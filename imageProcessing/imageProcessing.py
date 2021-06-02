from IPython import display

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time
import pickle
import os
from tensorflow import keras


class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )


    @tf.function
    def sample(self, eps=None):
    if eps is None:
        eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

    def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
        probs = tf.sigmoid(logits)
        return probs
    return logits


class ImageProcessing:
    """ This class will train the various models passed through it.
        It will save models it has trained into persistant memory and 
        load previously trained models. """

    def __init__(self):
        self.model = None
        self.my_path = os.path.abspath(__file__)
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def getModel(self):
        return self.model

    def setModel(self, model=None):
        if model == None:
            raise Exception("Model is set to `None`")
        self.model = model

    def CVAE_log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def CVAE_compute_loss(self, model, x):
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.CVAE_log_normal_pdf(z, 0., 0.)
        logqz_x = self.CVAE_log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)


    @tf.function
    def CVAE_train_step(self, model, x, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and uses the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = self.CVAE_compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def generate_and_save_images(self, test_sample, epoch=0, training_image=False):
        mean, logvar = self.model.encode(test_sample)
        z = self.model.reparameterize(mean, logvar)
        predictions = self.model.sample(z)
        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')

        # tight_layout minimizes the overlap between 2 sub-plots
        if(training_image == False):
            plt.savefig(os.path.join(self.my_path , 'savedImages', 'CVAE_image_at_epoch_{:04d}.png'.format(epoch)))
        else:
            plt.savefig(os.path.join(self.my_path ,'savedImages', 'training', 'CVAE_image_at_epoch_{:04d}.png'.format(epoch)))
            plt.show()

    def preprocess_images(self, images):
        images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
        return np.where(images > .5, 1.0, 0.0).astype('float32')

    def display_image(self, epoch_no):
        return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

    def train_CVAE(self, epochs = 10, model = None):
        """ This functions trains the CVAE model """
        latent_dim = 2
        if model == None:
            if self.model == None:
                # We use 2 because that is the latent dimensions we want to use
                self.model = CVAE(latent_dim)

            model = self.model

        # This function get the dataset from keras datasets
        # In the final implementation we will use have this as the default dataset
        # and also have the user import their own custom dataset
        (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

        train_images = self.preprocess_images(train_images)
        test_images = self.preprocess_images(test_images)

        train_size = 60000
        batch_size = 32
        test_size = 10000

        train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
        test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                        .shuffle(test_size).batch(batch_size))

        # epochs = 10
        # set the dimensionality of the latent space to a plane for visualization later
        num_examples_to_generate = 16

        # keeping the random vector constant for generation (prediction) so
        # it will be easier to see the improvement.
        random_vector_for_generation = tf.random.normal(
            shape=[num_examples_to_generate, latent_dim])

        # model = CVAE(latent_dim)

        # Pick a sample of the test set for generating output images
        assert batch_size >= num_examples_to_generate
        for test_batch in test_dataset.take(1):
            test_sample = test_batch[0:num_examples_to_generate, :, :, :]

        # self.generate_and_save_images(test_sample, 0, True)

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            print("Iteration: {}".format(epoch))
            for train_x in train_dataset:
                self.CVAE_train_step(model, train_x, self.optimizer)
            end_time = time.time()

            loss = tf.keras.metrics.Mean()
            for test_x in test_dataset:
                loss(self.CVAE_compute_loss(model, test_x))
            elbo = -loss.result()
            display.clear_output(wait=False)
            print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                    .format(epoch, elbo, end_time - start_time))
            # self.generate_and_save_images(test_sample, epoch, True)
        
        model.build(train_images.shape)
        print("Training complete")

class ModelEncapsulator:
    def __init__(self):
        self.models = []

    def getNumberOfModels(self):
        return len(self.models)

    def getModels(self):
        return models

    def getModel(self, index):
        if len(self.models) == 0:
            raise Exception("There are no available models")

        if index >= len(self.models):
            raise Exception("Index out of range available models")
        elif index < len(self.modle):
            raise Exception("Index out of range available models")
        
        return models[index]


    def save_model(self, model=None, file_name="./savedModels/CVAE_Model.pickle"):
        """ This method saves a trained model """
        if model == None:
            raise Exception("Model is set to `None`")
        else:
            store = model
            store.save(file_name)
            print("FILE SAVED")

    def load_model(self, file_name="./savedModels/CVAE_Model"):
        """ This method loads a saved and trained model """
        store = keras.models.load_model(file_name)
        if store == None:
            raise Exception("Model is set to `None`")
        else:
            print("FILE READ")
            self.models.append(store)
            return store
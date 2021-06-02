import tensorflow as tf
from imageProcessing import ImageProcessing, ModelEncapsulator

if __name__ == "__main__":
    
    initialModel = None
    initialModel = ImageProcessing()
    modelEncapsulator = ModelEncapsulator()

    initialModel.train_CVAE(2)
    modelEncapsulator.save_model(initialModel)
    print("Model saved!")
    anotherModel = modelEncapsulator.load_model()

    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

    test_images = initialModel.preprocess_images(test_images)

    batch_size = 32
    num_examples_to_generate = 16

    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                        .shuffle(10000).batch(batch_size))

    assert batch_size >= num_examples_to_generate
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]

    initialModel.generate_image_and_save_images(test_sample)
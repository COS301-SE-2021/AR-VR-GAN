from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torchvision import transforms
from torchvision.utils import save_image
from VAEModel import VAE
from ConvolutionalAutoencoder import ConvolutionalAutoencoder
from CVAEModel import CVAE
import os
from datetime import datetime, time
from PIL import Image
from modelExceptions import ModelException
from DataLoaders import DataLoaders

from os import listdir
from os.path import isfile, join

class ModelGenerator:
    """ This class trains VAE models and generates images from said models

    This class trains and saves VAE models. It also generates and saves images
    generated from said models. 
    
    Attributes
    ----------
    model : VAE
        a formatted string to print out what the animal says
    train_loader : torch.utils.data.DataLoader
        An object that stores the set of training images
    test_loader : torch.utils.data.DataLoader
        An object that stores the set of test images

    Methods
    -------
    loss_function()
        Produces the loss the of training iteration

    train_model(epochs)
        Trains and test the model over a set of iterations(epochs)

    train(epoch=int)
        Trains the model on the set of images in train_loader

    test(epoch=int, generate=bool)
        Tests the accuracy of the model with the set of images in test_loader

    loadModel(filepath=str)
        Loads a previously saved model to be used by the model generator

    saveModel(filepath=str)
        Saves the model currently held by the model generator 

    generateImage(vector=list, filepath=str)
        Generates an image from the model from the vector pass into the function

    clearModel()
        Removes the current model and replaces it with a new untrained model

    set_latent_size(latent_size=int)
        Sets the latent vector to a new size, specified by the user

    loadDataset()
        Loads a new dataset to train the model on 

    """

    def __init__(self) -> None:
        # Sets the seed for generating random numbers. Returns a torch.Generator object.
        torch.manual_seed(1)

        # A torch.device is an object representing the device on which a torch.
        # Tensor is or will be allocated. The torch.device contains a device type ('cpu' or 'cuda')
        # and optional device ordinal for the device type. If the device ordinal is not present, 
        # this object will always represent the current device for the device type, 
        # even after torch.cuda.set_device() is called.
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")

        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}

        # To create a custom dataset go to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        im_transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.ToTensor(),
        ])
        self.data_loaders = DataLoaders(im_transform, kwargs)
        
        self.model = None

    def train_model(self, epochs: int, latent_vector: int, dataset: str = "mnist", model_type: str = "cvae", 
    beta: int=1, name: str="", show: bool=False) -> None:
        """ Trains and test the model over a set of iterations(epochs)

        Parameters
        ----------
        epochs : int
            The number of iterations the model will be tested
        """
        self.set_latent_size(latent_vector)
        loader = self.data_loaders.get_dataloader(dataset)
        loader_iter = iter(loader)
        images, labels = loader_iter.next()
        channel_size = images.shape[1]

        if model_type == "convolutional":
            self.model = ConvolutionalAutoencoder(latent_vector, channel_size)
            self.model.training_loop(epochs, loader, name, False)
        elif model_type == "cvae":
            self.model = CVAE(channel_size, latent_vector)
            self.model.training_loop(epochs, loader, beta, name, False)
        else:
            self.model = CVAE(channel_size,latent_vector)
            self.model.training_loop(epochs, loader, beta, name, False)

    def loadModel(self, filepath: str="") -> str:
        """Loads a previously saved model to be used by the model generator

        If a file path is not specified then the model generator will load the default model.
        
        Parameters
        ----------
        filepath : str
            The destination of a saved model

        Raises
        ------
        ModelException
            If the destination of the file does not exist or 
            if the file path is not a pytorch file.
        """
        if filepath == "":
            filepath = "defaultModels/Epochs-50.pt"
            self.model = torch.load(filepath)
            print("Default VAE Model loaded")
            return filepath
        else:
            if not os.path.isfile(filepath):
                raise ModelException(f"File {filepath} does not exist")
            elif not filepath.endswith('.pt'):
                raise ModelException("File needs to be a pytorch file")
            else:
                self.model = torch.load(filepath)
                self.set_latent_size(self.model.retrieve_latent_size())
                print(filepath+" VAE Model loaded")
                return filepath

    def saveModel(self, filepath: str="") -> str:
        """Saves the model currently held by the model generator 

        If a file path is not specified or the file path already exists then the model generator 
        will give the model the default name of "VAE-MODEL-" with a the timestamp appended to the 
        file name.
        
        Parameters
        ----------
        filepath : str
            The destination of a saved model

        """
        modelPath = ""
        if filepath == "":
            modelPath = "savedModels/VAE-MODEL-"+datetime.now().strftime("%d%m%Y%H%M%S")+".pt"
            torch.save(self.model, modelPath)
            print("Model saved as savedModels/VAE-MODEL-"+datetime.now().strftime("%d%m%Y%H%M%S")+".pt")
            return modelPath 
        else:
            if not filepath.endswith('.pt'):
                filepath+=".pt"
            
            if os.path.isfile(filepath):
                modelPath = "savedModels/VAE-MODEL-"+datetime.now().strftime("%d%m%Y%H%M%S")+".pt"
                print(filepath, " already exists!")
                torch.save(self.model, modelPath)
                print("Model saved as "+modelPath)

                return modelPath
            else:
                torch.save(self.model, filepath)
                print("Model saved as " + filepath)
                return filepath
    
    def generateImage(self, vector: list, filepath: str="") -> list:
        """ Generates an image from the model from the vector pass into the function

        Parameters
        ----------
        vector: list
            The reparameterization vector specified by the user
        filepath : str
            The destination the image will be saved
        
        Raises
        ------
        ModelException
            If the input vector dimensions are not the same as the ones specified in the model
        """
        if filepath == "":
            filepath = "savedImages/"+datetime.now().strftime("%d%m%Y%H%M%S")+".png"

        if not filepath.endswith(('.png', '.jpg', '.jpeg')):
            ModelException("File extension must be either be png, jpg, jpeg")
        else:    
            with torch.no_grad():
                if self.model.retrieve_latent_size() != len(vector):
                    raise ModelException("Input vector not the same size as model's vector")

                sample = torch.tensor([vector]).to(self.device)
                sample = self.model.decoder(sample).cpu() 

                save_image(sample.view(1, sample.shape[1], 28, 28), filepath)
                image = Image.open(filepath)
                new_image = image.resize((400, 400))
                new_image.save(filepath)

                with open(filepath, "rb") as image:
                    f = image.read()
                    b = bytearray(f)

                # os.remove(filepath)
                return list(b)

    def clearModel(self) -> None:
        """Removes the current model and replaces it with a new untrained model
        """
        self.model = None
        self.model = VAE(self.latent_size).to(self.device)

    def set_latent_size(self, latent_size: int) -> None:
        self.latent_size = latent_size

    def get_available_models(self, default: bool = True):
        if default != True:
            mypath = "./savedModels/"
        else:
            mypath = "./defaultModels/"

        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        return onlyfiles

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model Generator Model')
    parser.add_argument('--latent-size', type=int, default=3, metavar='N',
                        help='Determines the size latent size of the model to be trained')
    parser.add_argument('--model', type=str, default="defaultModels/Epochs-50.pt", metavar='N',
                        help='Choose the pre-saved model')
    parser.add_argument('--coordinates', default=None, type=float, nargs='+',
                        help='the latent vector to be used by the model to generate an image')
    args = parser.parse_args()

    generator = ModelGenerator()
    generator.train_model(15, 3, "cifar10", model_type="cvae", name="Beta-1-CIFAR-15")
    generator.saveModel("savedModels/Beta-1-CIFAR-15.pt")
    generator.generateImage([0.0, 0.0, 0.0])

    generator = ModelGenerator()
    generator.train_model(15, 3, "cifar10", model_type="convolutional", name="Normal-CIFAR-15")
    generator.saveModel("savedModels/Beta-1-CIFAR-15.pt")
    generator.generateImage([0.0, 0.0, 0.0])


    generator = ModelGenerator()
    generator.train_model(15, 3, "mnist", model_type="cvae", name="Beta-1-MNIST-15")
    generator.saveModel("savedModels/Beta-1-MNIST-15.pt")
    generator.generateImage([0.0, 0.0, 0.0])

    generator = ModelGenerator()
    generator.train_model(15, 3, "mnist", model_type="convolutional", name="Normal-MNIST-15")
    generator.saveModel("savedModels/Normal-MNIST-15.pt")
    generator.generateImage([0.0, 0.0, 0.0])
    
    # generator.train_model(50, 5)
    # generator.saveModel("defaultModels/BetaVAE5-CIRA10-Epochs-50.pt")
    from time import sleep
    # generator.saveModel("savedModels/CBeta-1-MNIST-1.pt")
    # sleep(1)
    # generator.generateImage([0.1, 0.0, 0.0])
    # sleep(1)
    # generator.generateImage([0.2, 0.1, 0.0])
    # sleep(1)
    # generator.generateImage([0.03, 0.1, 0.0])
    # sleep(1)
    # generator.generateImage([0.04, 0.1, 0.0])
    # sleep(1)
    # generator.generateImage([0.05, 0.1, 0.0])
    # sleep(1)
    # generator.generateImage([0.06, 0.1, 0.0])
    # sleep(1)
    # generator.generateImage([0.022, 0.1, 0.0])
    # sleep(1)
    # generator.generateImage([0.024, 0.1, 0.0])
    # sleep(1)
    # generator.generateImage([0.056, 0.1, 0.0])
    # generator.loadModel(args.model)

    # generator.to_tensorflow("./defaultModels/pytorch/Epochs-50.pt")
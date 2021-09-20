from __future__ import print_function
import torch
import torch.utils.data
from torchvision import transforms
from torchvision.utils import save_image
from VAEModel import VAE as OGVAE
from ConvolutionalAutoencoder import CAutoencoder
from CVAE import VAE
import os
from datetime import datetime
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
    model : VAE or CVAE or CAutoencoder
        a formatted string to print out what the animal says
    data_loaders: DataLoaders
        An object that stores the set of datasets that can be used in the neural networks

    Methods
    -------
    train_model(epochs=int, latent_vector=int, dataset=mnist, model_type=str, beta=int, name=str)
        Trains and test the model over a set of iterations(epochs)

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
    
    get_available_models(default=bool)
        Returns a list of available models

    """

    def __init__(self, dataloaders = None) -> None:
        torch.manual_seed(1)
        self.cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")

        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}

        im_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if dataloaders == None:
            self.data_loaders = DataLoaders(im_transform, kwargs)
        else:
            self.data_loaders = dataloaders
        self.model = None

    def train_model(self, epochs: int, latent_vector: int, dataset: str = "mnist", model_type: str = "cvae", 
    beta: int=1, name: str="") -> None:
        """ Trains and test the model over a set of iterations(epochs)

        Parameters
        ----------
        epochs : int
            The number of iterations the model will be tested
        """
        self.set_latent_size(latent_vector)
        im_transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor(),
        ])
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.cuda else {}
        if dataset == "celeba":
            self.data_loaders = DataLoaders(im_transform, kwargs)

        loader = self.data_loaders.get_dataloader(dataset)
        loader_iter = iter(loader)
        images, labels = loader_iter.next()
        channel_size = images.shape[1]

        if model_type == "convolutional":
            if dataset != "cifar10":
                self.model = CAutoencoder(latent_vector, channel_size, images.shape[2])
            else:
                self.model = CAutoencoder(latent_vector, channel_size)
            self.model.datasetUsed = dataset
            self.model.name = name
            self.model.training_loop(epochs, loader)
        else :
            if dataset == "fashion" or dataset == "mnist":
                self.model = OGVAE(latent_vector)
            else:
                self.model = VAE(channel_size, latent_vector)
    
            self.model.datasetUsed = dataset
            self.model.name = name
            self.model.training_loop(epochs, loader, beta)      

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
            filepath = "defaultModels/Beta-1-MNIST-50.pt"
            if torch.cuda.is_available():
                self.model = torch.load(filepath)
            else:
                self.model = torch.load(filepath, map_location=torch.device('cpu'))
            print("Default VAE Model loaded")
            return filepath
        else:
            if not filepath.endswith('.pt'):
                raise ModelException("File needs to be a pytorch file")

            if filepath in self.get_available_models():
                filepath = "./defaultModels/"+filepath
                if torch.cuda.is_available():
                    self.model = torch.load(filepath)
                else:
                    self.model = torch.load(filepath, map_location=torch.device('cpu'))
            elif filepath in self.get_available_models(False):
                # self.model = torch.load("./savedModels/"+filepath)
                if torch.cuda.is_available():
                    self.model = torch.load("./savedModels/"+filepath)
                else:
                    self.model = torch.load("./savedModels/"+filepath, map_location=torch.device('cpu'))
            else:
                raise ModelException(f"File {filepath} does not exist")

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

                save_image(sample.view(1, sample.shape[1], sample.shape[2], sample.shape[3]), filepath)
                image = Image.open(filepath)
                new_image = image.resize((400, 400))
                new_image.save(filepath)

                with open(filepath, "rb") as image:
                    f = image.read()
                    b = bytearray(f)

                os.remove(filepath)
                return list(b)

    def clearModel(self) -> None:
        """Removes the current model and replaces it with a new untrained model
        """
        self.model = None
        self.model = VAE(self.latent_size).to(self.device)

    def set_latent_size(self, latent_size: int) -> None:
        self.latent_size = latent_size

    def get_available_models(self, default: bool = True) -> list:
        """This function returns a list of models that are either in `defaultModels` folder or 
        the `savedModels` folder"""
        if default != True:
            mypath = "./savedModels/"
        else:
            mypath = "./defaultModels/"

        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        return onlyfiles

if __name__ == "__main__":
    generator = ModelGenerator()
    # To create a new model run the function below
    # The first parameter is the number of epochs(iterations), the second is the size of the latent
    # vector (so its best you leave it a 3), the third is dataset you want to train it on, you can 
    # choose betweeen `cifar10`, `mnist`, and `fashion`, model_type can be left as blank it just 
    # changes between a Variational autoencoder and a normal autoencoder, beta can be set to any integer 
    # greater than or equal to one but there are no checks for this and `name` will be the name of the 
    # checkpoint of the model. I adjusted 
    # the training sequence so that it saves the model every 10 epochs so that you can cancel the 
    # training any time with out losing progress.
    # Remember to save it
    # example below
    # generator = ModelGenerator()
    # generator.train_model(100, 3, "fashion", model_type="cvae", name="Beta-1-Fashion-100", beta=1)
    # generator.save("your-name.pt")
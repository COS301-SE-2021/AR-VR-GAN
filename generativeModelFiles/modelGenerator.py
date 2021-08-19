from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from VAEModel import VAE
import os
from datetime import datetime
from PIL import Image
from modelExceptions import ModelException

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
    # TODO: Handle edge cases in function i.e when a model is not loaded raise an exception
    # TODO: Create a throughout main i.e one that handles executes all the functions 
    # TODO: Adjust test files so that it can handle file changes made 
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

        # Loads the MNIST dataset 
        # You can change from MNIST to any other torch dataset by changing
        # datsets.MNIST to datasets.Caltect 101
        # Go to https://pytorch.org/vision/stable/datasets.html for more information

        # To create a custom dataset go to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.ToTensor()),
            batch_size= 128, shuffle=True, **kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
            batch_size=128, shuffle=True, **kwargs)

        self.model = VAE(3).to(self.device)
        self.latent_size = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self,recon_x, x, mu, logvar) -> float:
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def train_model(self, epochs) -> None:
        """ Trains and test the model over a set of iterations(epochs)

        Parameters
        ----------
        epochs : int
            The number of iterations the model will be tested
        """

        # Check whether a valid integer is passed and pass through the fuction
        # if not throw an exception. Make sure that it is in fact an integer and
        # that it is greater than zero.
        for epoch in range(1, epochs + 1):
            self.train(epoch)
            self.test(epoch)

    def train(self, epoch) -> None:
        """Trains the model on the set of images in train_loader
        Parameters
        ----------
        epoch : int
            The iteration the training is currently executing
        """
        self.model.train()
        train_loss = 0
        # Note torch.DataLoader combines a dataset and a sampler, and provides an 
        # iterable over the given dataset. Therefore `batch_idx` gives an index of 
        # which image the function is on (it is an int), `data` provides the image 
        # (it is a tensor and its shape is torch.Size([128, 1, 28, 28])) and `_` is
        #  the sampler (also a tensor and it shape is torch.Size([128])). 
        # Note that 128 is a common number because that is the batch size of the dataset
        # And 28 is the height and width of the image in pixels. 
        # From Medium https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4: 
        # All images are size normalized to fit in a 20x20 pixel box and there are centered in a 28x28 image using the center of mass.
        for batch_idx, (data, _) in enumerate(self.train_loader):
            # To view tensor in text file (Note returns a LARGE array of float values
            # which supposed to represent each pixel in an image )
            # numpy.savetxt('my_file.txt', data.numpy().reshape(4,-1))

            # Converts `data` to a tensor with a specified dtype
            data = data.to(self.device)
            # Info on this https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch0
            self.optimizer.zero_grad()
            # This returns the output from forward in VAE model
            recon_batch, mu, logvar = self.model(data)
            # `recon_batch` returns all 128 images reconstructed
            # The shape is torch.Size([128, 784]) implying that there are 128 images and
            # the dimensions have been reduced to a single array since sqrt(784) and 
            # earlier it is stated that the dimensions of an mnist image is 28x28.
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            # Displays training data
            # if batch_idx % 10 == 0: # Change the 10 to the log_interval (self.args.log_interval)
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, batch_idx * len(data), len(self.train_loader.dataset),
                #     100. * batch_idx / len(self.train_loader),
                #     loss.item() / len(data)))

        # print('====> Epoch: {} Average loss: {:.4f}'.format(
        #     epoch, train_loss / len(self.train_loader.dataset)))

    def test(self, epoch, generate = False) -> None:
        """Tests the accuracy of the model with the set of images in test_loader
        If generate is set to True then the function will produce a comparison image 
        comparing the test data compared the image by the model
        Parameters
        ----------
        epoch : int
            The iteration the training is currently executing
        generate : bool, optional
            To decide whether or not to save an image displaying the comparison between the model and the actual results
        """
        # Note to self change 128 to batch size and 28 to the pixels
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n],
                                        recon_batch.view(128, 1, 28, 28)[:n]])# Batch size instead of 128
                    if generate == True:
                        save_image(comparison.cpu(),
                                'test/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(self.test_loader.dataset)
        # print('====> Test set loss: {:.4f}'.format(test_loss))

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
            self.model = torch.load("defaultModels/Epochs-50.pt")
            print("Default VAE Model loaded")
            return filepath
            # return True
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
                # return True

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
                print("Model saved as" + filepath)
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
                # Returns a 1x20 array with random values from 0-1
                # The second value in the randn decides the the dimesion check VAEModel
                # sample = torch.randn(1,2).to(self.device)
                if self.model.retrieve_latent_size() != len(vector):
                    raise ModelException("Input vector not the same size as model's vector")

                sample = torch.tensor([vector]).to(self.device)
                # print(sample)
                # print(sample.size())
                sample = self.model.decode(sample).cpu() 
                save_image(sample.view(1, 1, 28, 28), filepath)
                image = Image.open(filepath)
                new_image = image.resize((400, 400))
                new_image.save(filepath)

                #print("Imaged Saved as "+filepath)
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

    def loadDataset(self) -> None:
        # TODO Complete this function but first create dataset loader class
        #       and related variables that need to created in this class.
        """Loads a new dataset to train the model on 
        """
        pass


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
    generator.loadModel(args.model)

    # generator.to_tensorflow("./defaultModels/pytorch/Epochs-50.pt")
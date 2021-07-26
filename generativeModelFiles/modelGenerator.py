from __future__ import print_function
import argparse
import numpy
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import sampler
from torchvision import datasets, transforms
from torchvision.utils import save_image
from VAEModel import VAE
import os
from datetime import datetime
from PIL import Image
from modelExceptions import ModelException

class ModelGenerator:
    def __init__(self):

        # This section dealt with taking in default arguements in the orginal VAE MNIST Example
        # This section is unecessary for AR-VR-GAN, we just have to replace the arguement values
        # in the appropriate places.
        self.parser = argparse.ArgumentParser(description='VAE MNIST Example')
        self.parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='input batch size for training (default: 128)')
        self.parser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
        self.parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        self.parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        self.parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
        self.args = self.parser.parse_args()
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()

        # Sets the seed for generating random numbers. Returns a torch.Generator object.
        torch.manual_seed(self.args.seed)

        # A torch.device is an object representing the device on which a torch.
        # Tensor is or will be allocated. The torch.device contains a device type ('cpu' or 'cuda')
        # and optional device ordinal for the device type. If the device ordinal is not present, 
        # this object will always represent the current device for the device type, 
        # even after torch.cuda.set_device() is called.
        self.device = torch.device("cuda" if self.args.cuda else "cpu")

        kwargs = {'num_workers': 1, 'pin_memory': True} if self.args.cuda else {}

        # Loads the MNIST dataset 
        # You can change from MNIST to any other torch dataset by changing
        # datsets.MNIST to datasets.Caltect 101
        # Go to https://pytorch.org/vision/stable/datasets.html for more information

        # To create a custom dataset go to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.ToTensor()),
            batch_size= self.args.batch_size, shuffle=True, **kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
            batch_size=self.args.batch_size, shuffle=True, **kwargs)

        self.model = VAE(3).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self,recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def train_model(self, epochs):
        for epoch in range(1, epochs + 1):
            self.train(epoch)
            self.test(epoch)

    def train(self, epoch):
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
            # print(f"Batch IDX TYPE: {type(batch_idx)}")
            # print(f"Batch IDX: {batch_idx}")
            # print(f"DATA TYPE: {type(data)}")
            # To view tensor in text file (Note returns a LARGE array of float values
            # which supposed to represent each pixel in an image )
            # numpy.savetxt('my_file.txt', data.numpy().reshape(4,-1))
            # print(f"DATA: {data.size()}")
            # print(f"Underscore TYPE: {type(_)}")
            # print(f"Underscore: {_.size()}")


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
            # print(recon_batch.size())
            # print("="*10)
            # print(mu.size())
            # print("="*10)
            # print(logvar.size())
            # print("="*10)
            # input()
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(self.train_loader.dataset)))

    def test(self, epoch):
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
                                        recon_batch.view(self.args.batch_size, 1, 28, 28)[:n]])
                    # save_image(comparison.cpu(),
                    #         'test/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def loadModel(self, filepath=""):
        if filepath == "":
            self.model = torch.load("defaultModels/Epochs-50.pt")
            print("Default VAE Model loaded")
            return True
        else:
            if not os.path.isfile(filepath):
                raise ModelException(f"File {filepath} does not exist")
            elif not filepath.endswith('.pt'):
                raise ModelException("File needs to be a pytorch file")
            else:
                self.model = torch.load(filepath)
                print(filepath+" VAE Model loaded")
                return True

    def saveModel(self, filepath=""):
        modelPath = ""
        if filepath == "":
            modelPath = "savedModels/VAE-MODEL-"+datetime.now().strftime("%d%m%Y%H%M%S")+".pt"
            torch.save(self.model, modelPath)
            print("Model saved as savedModels/VAE-MODEL-"+datetime.now().strftime("%d%m%Y%H%M%S")+".pt")
            return True
        else:
            if not filepath.endswith('.pt'):
                filepath+=".pt"
            
            if os.path.isfile(filepath):
                modelPath = "savedModels/VAE-MODEL-"+datetime.now().strftime("%d%m%Y%H%M%S")+".pt"
                print(filepath, " already exists!")
                torch.save(self.model, modelPath)
                print("Model saved as savedModels/VAE-MODEL-"+modelPath)
                return True
            else:
                torch.save(self.model, filepath)
                print("Model saved as" + filepath)
                return True
    
    def generateImage(self, vector = None, filepath=""):
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
                print(sample)
                print(sample.size())
                sample = self.model.decode(sample).cpu() 
                save_image(sample.view(1, 1, 28, 28), filepath)
                image = Image.open(filepath)
                new_image = image.resize((400, 400))
                new_image.save(filepath)
                print("Imaged Saved as "+filepath)
                return True

    def clearModel(self):
        self.model = None
        self.model = VAE().to(self.device)

    def set_latent_size(self, latent_size):
        self.latent_size = latent_size


    def loadDataset(self):
        pass


if __name__ == "__main__":
    generator = ModelGenerator()
    generator.loadModel("savedModels/50iterations.pt")
    # generator.train_model(50)
    # generator.saveModel("savedModels/50iterations.pt")
    generator.generateImage([0.000000, 0.000, 0.00])
    # print(generator.test_loader)
    # generator.model.encode()
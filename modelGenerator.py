from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from VAEModel import VAE
import os
from datetime import datetime
from PIL import Image

class ModelGenerator:
    def __init__(self):
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

        torch.manual_seed(self.args.seed)

        self.device = torch.device("cuda" if self.args.cuda else "cpu")

        kwargs = {'num_workers': 1, 'pin_memory': True} if self.args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.ToTensor()),
            batch_size= self.args.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
            batch_size=self.args.batch_size, shuffle=True, **kwargs)

        self.model = VAE().to(self.device)
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


    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
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
                    save_image(comparison.cpu(),
                            'results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def loadModel(self, filepath=""):
        if filepath == "":
            self.model = torch.load("defaultModels/Epochs-50.pt")
            print("Default VAE Model loaded")
            return True
        else:
            if not os.path.isfile(filepath):
                raise Exception(f"File {filepath} does not exist")
            elif not filepath.endswith('.pt'):
                raise Exception(f"File needs to be a pytorch file")
            else:
                self.model = torch.load(filepath)
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
                torch.save(self.model, modelPath)
                print(filepath, " already exists!")
                print("Model saved as savedModels/VAE-MODEL-"+datetime.now().strftime("%d%m%Y%H%M%S")+".pt")
            else:
                self.model = torch.save(self.model, filepath)
                print("Model saved as" + filepath)
                return True
    
    def generateImage(self, filepath=""):
        if filepath == "":
            filepath = "savedImages/"+datetime.now().strftime("%d%m%Y%H%M%S")+".png"
        if not os.path.isfile(filepath):
            raise Exception(f"File {filepath} does not exist")
        else:
            if not filepath.endswith(('.png', '.jpg', '.jpeg')):
                Exception("File extension must be either be png, jpg, jpeg")
            else:    
                with torch.no_grad():
                    # Returns a 1x20 array with random values from 0-1
                    sample = torch.randn(1,20).to(self.device)
                    sample = self.model.decode(sample).cpu() 
                    save_image(sample.view(1, 1, 28, 28), filepath)
                    image = Image.open(filepath)
                    new_image = image.resize((400, 400))
                    new_image.save(filepath)
                    print("Imaged Saved as "+filepath)
                    return True
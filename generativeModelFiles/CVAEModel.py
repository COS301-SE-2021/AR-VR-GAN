from matplotlib import pyplot as plt
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn import functional as F
from torchvision.utils import save_image

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class CVAE(nn.Module):
    def __init__(self, channels: int = 3,latent_vector: int = 3):
        super().__init__()        
        # input is Nx1x28x28
        # N, 3, 28, 28
        self.z = latent_vector
        self.channels = channels
        self.encoder = nn.Sequential(
            # Note we increase the channels but reduce the size of the image
            nn.Conv2d(channels, 16, 11, stride=1, padding=1), # -> (N, 16 , (14, 14) <- image size);<-output, reduces image size by half; 1<- input size, 16 <- output channels, 3 <- kernal size, 
            nn.ReLU(),
            nn.Conv2d(16, 32, 7, stride=1, padding=1), # -> N, 32, 7, 7;<-output ;16 <- input
            nn.ReLU(),
            nn.Conv2d(32, 64, 9, stride=1, padding=1), # -> N, 64, 1, 1
            nn.ReLU(),
            nn.Conv2d(64, 128, 12, stride=1, padding=1),
            View((-1, 128*1*1)), 
            nn.Linear(128, self.z*2),
            nn.Sigmoid()
        )
        
        # N , latent_vector, <- input
        self.decoder = nn.Sequential(
            nn.Linear(self.z, 128),
            View((-1, 128, 1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 12, stride=1, padding=1), # -> N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 9, stride=1, padding=1), # N, 16, 14, 14 (N,16,13,13 without output_padding)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 7, stride=1, padding=1), # N, 1, 28, 28  (N,1,27,27)
            nn.ReLU(),
            nn.ConvTranspose2d(16, channels, 11, stride=1, padding=1), # N, 1, 28, 28  (N,1,27,27)
            nn.Sigmoid()
        )

    def forward(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.z]
        logvar = distributions[:, self.z:]
        z = self.reparametrize(mu, logvar)
        x_recon = self.decoder(z)

        return x_recon, mu , logvar
    
    def reparametrize(self, mu, logvar):
        # print("logvar",logvar.shape)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        # print("std",std.shape)
        # print("eps", eps.shape)
        # print("mu", mu)
        # input()
        return mu + std*eps

    def loss_function(self,recon_x, x, mu, logvar, beta = 1) -> float:
        BETA = beta
        # print(x.shape)
        # input()
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum').div(x.shape[0])
        # print("BCE", BCE)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + BETA*KLD

    def training_loop(self, epochs: int, train_loader, beta: int=1, name: str="", show: bool=False ) -> None:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.train()
        train_loss = 0
        outputs = []
        save = True
        if name == "" : 
            save == False
        
        j = 0
        for iter in range(epochs):
            i = 0
            j +=1
            for (data, _) in train_loader:
                i+=1
                recon_batch, mu, logvar = self(data)
                optimizer.zero_grad()
                loss = self.loss_function(recon_batch, data, mu, logvar, beta)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                # Displays training data
                if show == True:
                    save_image(recon_batch.view(recon_batch.shape[0], recon_batch.shape[1], 28, 28), f"./training/{name}-{iter+1}.png")
                print(f'{i}:Epoch:{iter+1}, Loss:{loss.item():.4f}')
                outputs.append((iter, data, recon_batch))
                if i == 350 : break
            if save and j % 10 == 0:
                torch.save(self, f"./savedModels/CVAE/{name}.pt")
    

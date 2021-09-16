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
        # input is Nx1x64x64
        # N, channels, 64, 64
        self.z = latent_vector
        self.beta = 1
        self.epochs = 0
        self.datasetUsed = ''
        self.name = ''

        self.channels = channels
        self.encoder = nn.Sequential(
            # Note we increase the channels but reduce the size of the image
            nn.Conv2d(channels, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, self.z*2),   
            # nn.Sigmoid()
        )
        
        # N , latent_vector, <- input
        self.decoder = nn.Sequential(
            nn.Linear(self.z, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, channels, 4, 2, 1),  # B, nc, 64, 64
            # nn.Sigmoid()
        )

    def forward(self, x):
        distributions = self.encoder(x)
        mu = distributions[:, :self.z]
        logvar = distributions[:, self.z:]
        z = self.reparametrize(mu, logvar)
        x_recon = self.decoder(z).view(x.size())

        return x_recon, mu , logvar
    
    def retrieve_latent_size(self) -> int:
        return self.z

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + std*eps

    def loss_function(self,recon_x, x, mu, logvar, beta = 1) -> float:
        BETA = beta
        BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum').div(x.shape[0])
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + BETA*KLD

    def training_loop(self, epochs: int, train_loader, beta: int=1, name: str="", show: bool=False ) -> None:
        self.beta = beta
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.train()
        train_loss = 0
        outputs = []
        save = True
        if name == "" :
            name = "no-name" 
            save == False
        
        self.name = name
        for iter in range(epochs):
            i = 0
            self.epochs +=1
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
                    save_image(recon_batch.view(recon_batch.shape[0], recon_batch.shape[1], recon_batch.shape[2], recon_batch.shape[3]), f"./training/{name}-{iter+1}.png")
                print(f'{i}:Epoch:{iter+1}, Loss:{loss.item():.4f}')
                outputs.append((iter, data, recon_batch))
                if i == 500 : break
            if save and self.epochs % 10 == 0:
                torch.save(self, f"./savedModels/CVAE/{name}.pt")
    
    def details(self) -> dict:
        model_details: dict = {
            "name": self.name,
            "epochs_trained": self.epochs,
            "latent_vector_size": self.z,
            "beta_value": self.beta,
            "dataset_used": self.datasetUsed
        }

        return model_details
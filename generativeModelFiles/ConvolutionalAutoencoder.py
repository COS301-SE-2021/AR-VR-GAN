from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class CAutoencoder(nn.Module):
    def __init__(self, latent_vector: int = 3, channel_size: int=3):
        super().__init__()        
        # input is Nx1x64x64
        # N, 3, 64, 64
        self.z = latent_vector
        self.epochs = 0
        self.datasetUsed = ''

        self.channel_size = channel_size
        self.encoder = nn.Sequential(
            # Note we increase the channels but reduce the size of the image
            nn.Conv2d(channel_size, 32, 4, 2, 1),          # B,  32, 32, 32
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
            nn.Linear(256, self.z), 
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
            nn.ConvTranspose2d(32, channel_size, 4, 2, 1),  # B, nc, 64, 64
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def retrieve_latent_size(self) -> int:
        return self.z

    def training_loop(self, epochs: int, train_loader, name: str="", show: bool=False ) -> None:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.train()

        outputs = []
        save = False
        if name == "" : 
            name = "no-name" 
            save == True
        
        for iter in range(epochs):
            i = 0
            self.epochs +=1
            for (data, _) in train_loader:
                i+=1
                recon = self(data)
                loss = criterion(recon,data)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if show == True:
                    save_image(recon.view(recon.shape[0], recon.shape[1], recon.shape[2], recon.shape[3]), f"./training/{name}-{iter+1}.png")
                print(f'{i}:Epoch:{iter+1}, Loss:{loss.item():.4f}')
                outputs.append((iter, data, recon))
                # if i == 350 : break
            if save and self.epochs % 10 == 0:
                torch.save(self, f"./savedModels/CAE/{name}.pt")

    def details(self):
        model_details: dict = {
            "name": self.name,
            "epochs_trained": self.epochs,
            "latent_vector_size": self.z,
            "dataset_used": self.datasetUsed
        }

        return model_details

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

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
            nn.Linear(128, self.z)
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
            # nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

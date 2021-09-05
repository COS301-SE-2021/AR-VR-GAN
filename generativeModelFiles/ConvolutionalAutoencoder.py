from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, latent_vector: int = 3):
        super().__init__()        
        # input is Nx1x28x28
        # N, 1, 28, 28
        last_filter: int = -(latent_vector-1-2-10)

        self.encoder = nn.Sequential(
            # Note we increase the channels but reduce the size of the image
            nn.Conv2d(3, 16, 11, stride=1, padding=1), # -> (N, 16 , (14, 14) <- image size);<-output, reduces image size by half; 1<- input size, 16 <- output channels, 3 <- kernal size, 
            nn.ReLU(),
            nn.Conv2d(16, 32, 7, stride=1, padding=1), # -> N, 32, 7, 7;<-output ;16 <- input
            nn.ReLU(),
            nn.Conv2d(32, 64, 9, stride=1, padding=1), # -> N, 64, 1, 1
            nn.ReLU(),
            nn.Conv2d(64, 128, last_filter, stride=1, padding=1)
        )
        
        # N , 64, 1, 1 <- input
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7), # -> N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # N, 16, 14, 14 (N,16,13,13 without output_padding)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), # N, 1, 28, 28  (N,1,27,27)
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
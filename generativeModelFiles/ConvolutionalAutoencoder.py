from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

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
        self.name = ''

        self.channel_size = channel_size
        self.conv1 = nn.Conv2d(channel_size, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(8 * 8 * 16, 512)
        self.fc_bn1 = nn.BatchNorm1d(512)
        self.fc21 = nn.Linear(512, latent_vector)
        self.fc22 = nn.Linear(512, latent_vector)

        # Decoder
        self.fc3 = nn.Linear(latent_vector, 512)
        self.fc_bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 8 * 8 * 16)
        self.fc_bn4 = nn.BatchNorm1d(8 * 8 * 16)

        self.conv5 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, channel_size, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decode(encoded)
        return decoded

    def retrieve_latent_size(self) -> int:
        return self.z

    def encoder(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, 8 * 8 * 16)

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        return self.fc21(fc1), self.fc22(fc1)

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3))).view(-1, 16, 8, 8)

        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        return self.conv8(conv7).view(-1, self.channels, 32, 32)

    def decoder(self, z):
        fc3 = self.relu((self.fc3(z)))
        fc4 = self.relu((self.fc4(fc3))).view(-1, 16, 8, 8)

        conv5 = self.relu((self.conv5(fc4)))
        conv6 = self.relu((self.conv6(conv5)))
        conv7 = self.relu((self.conv7(conv6)))
        return self.conv8(conv7).view(-1, self.channels, 32, 32)


    def training_loop(self, epochs: int, train_loader, name: str="") -> None:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.train()
        
        for iter in range(epochs):
            i = 0
            self.epochs +=1
            loss_list = []
            print("epoch {}...".format(iter))
            for (data, _) in train_loader:
                i+=1
                if torch.cuda.is_available():
                    data = data.cuda()
                recon = self(data)
                loss = criterion(recon,data)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.data)
            print("epoch {}: - loss: {}".format(iter, np.mean(loss_list)))

    def details(self):
        model_details: dict = {
            "name": self.name,
            "epochs_trained": self.epochs,
            "latent_vector_size": self.z,
            "dataset_used": self.datasetUsed
        }

        return model_details

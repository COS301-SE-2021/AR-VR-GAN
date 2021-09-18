from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
from torch import optim
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader

class VAE(nn.Module):
    def __init__(self, channels: int = 3,latent_vector: int = 3):
        super(VAE, self).__init__()

        self.z = latent_vector
        self.beta = 1
        self.epochs = 0
        self.datasetUsed = ''
        self.name = ''
        self.channels = channels
        # Encoder
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
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
        self.conv8 = nn.ConvTranspose2d(16, channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU()

    def encode(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        conv2 = self.relu(self.bn2(self.conv2(conv1)))
        conv3 = self.relu(self.bn3(self.conv3(conv2)))
        conv4 = self.relu(self.bn4(self.conv4(conv3))).view(-1, 8 * 8 * 16)

        fc1 = self.relu(self.fc_bn1(self.fc1(conv4)))
        return self.fc21(fc1), self.fc22(fc1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

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

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def details(self) -> dict:
        model_details: dict = {
            "name": self.name,
            "epochs_trained": self.epochs,
            "latent_vector_size": self.z,
            "beta_value": self.beta,
            "dataset_used": self.datasetUsed
        }

        return model_details

    def retrieve_latent_size(self) -> int:
        return self.z

    def loss_function(self, recon_x, x, mu, logvar, beta=1):
        mse_loss = nn.MSELoss(size_average=False)
        MSE = mse_loss(recon_x, x)

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + beta*KLD

    def training_loop(self, epochs: int, train_loader, beta: int=1) -> None:
        self.train()
        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if self.cuda else "cpu")

        self.beta = beta
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        for epoch in range(0, epochs):
            loss_list = []
            self.epochs += 1
            print("epoch {}...".format(epoch))
            for batch_idx, (data, _) in enumerate(train_loader):
                if cuda:
                    data = data.cuda()
                data = Variable(data)
                optimizer.zero_grad()
                recon_batch, mu, logvar = self(data)
                loss = self.loss_function(recon_batch, data, mu, logvar, beta)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.data)

            print("epoch {}: - loss: {}".format(epoch, np.mean(loss_list)))
            # save_image(recon_batch.view(-1, 1, 32, 32), f"training/testingMNIST{epoch}.png")
                


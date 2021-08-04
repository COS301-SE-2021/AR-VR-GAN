from __future__ import print_function
import torch
from torch.functional import Tensor
from torch.types import Number
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, latent_size: int = 1) -> None:
        super(VAE, self).__init__()
        self.latent_size: int = latent_size
        # Applies a linear transformation to the incoming data: y = xA^T + by=xAT+b
        self.fc1: nn.Linear = nn.Linear(784, 400)
        self.fc21: nn.Linear = nn.Linear(400, latent_size) # Orginally 400 by 200
        self.fc22: nn.Linear = nn.Linear(400, latent_size)
        self.fc3: nn.Linear = nn.Linear(latent_size, 400)
        self.fc4: nn.Linear = nn.Linear(400, 784)

    def set_latent_size(self, latent_size: int) -> None:
        self.latent_size = latent_size
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, self.latent_size) # Orginally 400 by 200
        self.fc22 = nn.Linear(400, self.latent_size)
        self.fc3 = nn.Linear(self.latent_size, 400)
        self.fc4 = nn.Linear(400, 784)

    def retrieve_latent_size(self) -> int:
        return self.latent_size

    def encode(self, x) -> tuple:
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar) -> float:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std 

    def decode(self, z) -> Tensor:
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x) -> tuple:
        # x.view: Returns a new tensor with the same data as the self tensor but of a different shape
        # the size -1 is inferred from other dimensions
        # From ouptut this becomes a 128x784
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

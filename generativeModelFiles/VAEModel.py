from __future__ import print_function
import torch
from torch.functional import Tensor
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
import torch.optim as optim

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

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self,recon_x, x, mu, logvar, beta = 1) -> float:
        BETA = beta
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
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
        save = False
        if name == "" : 
            save == True
        
        j = 0
        for iter in range(epochs):
            i = 0
            j +=1
            for (data, _) in train_loader:
                i+=1
                optimizer.zero_grad()
                recon_batch, mu, logvar = self(data)
                loss = self.loss_function(recon_batch, data, mu, logvar, beta)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                # Displays training data
                if show == True:
                    save_image(recon_batch.view(recon_batch.shape[0], recon_batch.shape[1], 28, 28), f"./training/{name}-{iter+1}.png")
                print(f'{i}:Epoch:{iter+1}, Loss:{loss.item():.4f}')
                outputs.append((iter, data, recon_batch))
            if save and j % 10 == 0:
                torch.save(self, f"./savedModels/VAE/{name}.pt")
from __future__ import print_function
import torch
from torch.functional import Tensor
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image
import torch.optim as optim
import numpy as np

class VAE(nn.Module):
    def __init__(self, latent_size: int = 1) -> None:
        super(VAE, self).__init__()
        self.latent_size: int = latent_size
        # self.z = latent_vector
        self.beta = 1
        self.epochs = 0
        self.datasetUsed = ''
        self.name = ''

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

    def decoder(self, z) -> Tensor:
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3)).view(h3.shape[0], 1, 28, 28)

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

    def details(self) -> dict:
        model_details: dict = {
            "name": self.name,
            "epochs_trained": self.epochs,
            "latent_vector_size": self.latent_size,
            "beta_value": self.beta,
            "dataset_used": self.datasetUsed
        }

        return model_details

    def training_loop(self, epochs: int, train_loader, beta: int=1 ) -> None:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        cuda = torch.cuda.is_available()
        self.train()
        train_loss = 0
        self.beta = beta
        for iter in range(epochs):
            loss_list = []
            print("epoch {}...".format(iter))
            i = 0
            for (data, _) in train_loader:
                i+=1
                optimizer.zero_grad()
                if cuda:
                    data = data.cuda()
                recon_batch, mu, logvar = self(data)
                loss = self.loss_function(recon_batch, data, mu, logvar, beta)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                loss_list.append(loss.data)
            print("epoch {}: - loss: {}".format(iter, np.mean(loss_list)))
            # save_image(recon_batch.view(recon_batch.shape[0], 1, 28, 28), f"./training/OGMNIST-{iter}.png")
            
if __name__ == "__main__":
    from torchvision import datasets, transforms
    cuda = torch.cuda.is_available()
    model = VAE(3)
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    mnist = torch.utils.data.DataLoader( datasets.MNIST("../data",train=True, download=False, transform=transforms.ToTensor()),
            batch_size= 5, shuffle=True, **kwargs)

    # print("Hello")
    # input()
    # model.training_loop(1, mnist, 1)
    # sample = torch.tensor([[0.1, 0.0, 0.0]])
    # sample = model.decoder(sample).cpu()
    # sample.view(1, 1, 28,28)
    # print(sample.shape)
    # input()
    # save_image(sample.view(1, 1, 28, 28), "./OGMnistTest1.png")

    # sample = torch.tensor([[0.2, 0.1, 0.0]])
    # sample = model.decoder(sample).cpu()

    # save_image(sample.view(1, 1, 28, 28), "./OGMnistTest2.png")
    # sample = torch.tensor([[0.53, 0.3, 0.2]])
    # sample = model.decoder(sample).cpu()
    # save_image(sample.view(1, 1, 28, 28), "./OGMnistTest3.png")
    # sample = torch.tensor([[0.04, 0.1, 1.0]])
    # sample = model.decoder(sample).cpu()
    # save_image(sample.view(1, 1, 28, 28), "./OGMnistTest4.png")
    # sample = torch.tensor([[0.05, 0.1, 0.0]])
    # sample = model.decoder(sample).cpu()
    # save_image(sample.view(1, 1, 28, 28), "./OGMnistTest5.png")
    # sample = torch.tensor([[0.06, 0.1, 0.0]])
    # sample = model.decoder(sample).cpu()
    # save_image(sample.view(1, 1, 28, 28), "./OGMnistTest6.png")
    # sample = torch.tensor([[0, 0, 1.99999999999999999]])
    # sample = model.decoder(sample).cpu()
    # save_image(sample.view(1, 1, 28, 28), "./OGMnistTest7.png")
    # sample = torch.tensor([[0.0, 0.0, 0.0]])
    # sample = model.decoder(sample).cpu()
    # save_image(sample.view(1, 1, 28, 28), "./OGMnistTest8.png")
    # sample = torch.tensor([[50.0, 0.0, 99.0]])
    # sample = model.decoder(sample).cpu()
    # save_image(sample.view(1, 1, 28, 28), "./OGMnistTest9.png")
    # sample = torch.tensor([[0.056, 0.0000000000000, 0.15530333333333333333333]])
    # sample = model.decoder(sample).cpu()
    # save_image(sample.view(1, 1, 28, 28), "./OGMnistTest10.png")
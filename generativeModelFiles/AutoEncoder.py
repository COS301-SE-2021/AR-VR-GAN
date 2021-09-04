import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
#transforms.ToTensor
# Convert a PIL Image or numpy.ndarray to tensor. This transform does not support torchscript.
# Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor 
# of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes 
# (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8
# In the other cases, tensors are returned without scaling.

class Autoencoder(nn.Module):
    def __init__(self, latent_vector: int):
        super().__init__()
        # Encoder function
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_vector)
        )

        # Decoder function
        self.decoder = nn.Sequential(
            nn.Linear(latent_vector, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
        )

        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

if __name__ == "__main__":
    torch.manual_seed(1)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    im_transform = transforms.Compose([
                transforms.Resize((28,28)),
                transforms.ToTensor(),
        ])
    train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("../data", train=True, download=False, transform=im_transform),
    batch_size= 128, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10("../data", train=False, transform=im_transform),
    batch_size=128, shuffle=True, **kwargs)

    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for batch_idx, (data, _) in enumerate(train_loader):
            # To view tensor in text file (Note returns a LARGE array of float values
            # which supposed to represent each pixel in an image )
            # numpy.savetxt('my_file.txt', data.numpy().reshape(4,-1))

            # Converts `data` to a tensor with a specified dtype
            data = data.to(device)
            # Info on this https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch0
            optimizer.zero_grad()
            # This returns the output from forward in VAE model
            recon_batch, mu, logvar = model(data)
            # `recon_batch` returns all 128 images reconstructed
            # The shape is torch.Size([128, 784]) implying that there are 128 images and
            # the dimensions have been reduced to a single array since sqrt(784) and 
            # earlier it is stated that the dimensions of an mnist image is 28x28.
            loss = model.loss_function(recon_batch, data, mu, logvar, beta)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            # Displays training data
            if batch_idx % 10 == 0: # Change the 10 to the log_interval (self.args.log_interval)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)))
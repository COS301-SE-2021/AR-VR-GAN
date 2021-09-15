from matplotlib import pyplot as plt
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

    def loss_function(self, recon, x):
        criterion = nn.MSELoss()
        return criterion(recon, x)

    def training_loop(self, epochs: int, train_loader, beta: int=1, name: str="", show: bool=False ):
        model = self.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        save = False
        if name == "" : 
            save == True
        outputs = []
        j =0
        for epoch in range(epochs):
            j+=1
            for (data, _) in train_loader:
                img = data.reshape(-1, 28*28) # -> use for Autoencoder_Linear
                recon = model(img)
                loss = criterion(recon, img)
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
                outputs.append((epoch, img, recon))  
            if save and j % 10 == 0:
                torch.save(self, f"./savedModels/Autoencoder/{name}.pt")      

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

    model = Autoencoder(3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    num_epochs = 10
    outputs = []
    for epoch in range(num_epochs):
        for (data, _) in train_loader:
            # Converts `data` to a tensor with a specified dtype
            # data = data.to(device)
            # Info on this https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch0
            img = data.reshape(-1, 28*28) # -> use for Autoencoder_Linear
            recon = model(img)
            loss = criterion(recon, img)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch:{epoch+1}, Loss:{loss.item():.4f}')
            outputs.append((epoch, img, recon))

    for k in range(0, num_epochs):
        plt.figure(figsize=(9, 2))
        # plt.gray()
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, i+1)
            item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
            # item: 1, 28, 28
            plt.imshow(item[0])
                
        for i, item in enumerate(recon):
            if i >= 9: break
            plt.subplot(2, 9, 9+i+1) # row_length + i + 1
            item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
            # item: 1, 28, 28
            plt.imshow(item[0])
        plt.show()
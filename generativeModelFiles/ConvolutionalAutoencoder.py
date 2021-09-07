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


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, latent_vector: int = 3):
        super().__init__()        
        # input is Nx1x28x28
        # N, 3, 28, 28
        self.z = latent_vector
        self.encoder = nn.Sequential(
            # Note we increase the channels but reduce the size of the image
            nn.Conv2d(3, 16, 11, stride=1, padding=1), # -> (N, 16 , (14, 14) <- image size);<-output, reduces image size by half; 1<- input size, 16 <- output channels, 3 <- kernal size, 
            nn.ReLU(),
            nn.Conv2d(16, 32, 7, stride=1, padding=1), # -> N, 32, 7, 7;<-output ;16 <- input
            nn.ReLU(),
            nn.Conv2d(32, 64, 9, stride=1, padding=1), # -> N, 64, 1, 1
            nn.ReLU(),
            nn.Conv2d(64, 128, 12, stride=1, padding=1),
            View((-1, 128*1*1)), 
            nn.Linear(128, self.z),
            nn.Sigmoid()
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
            nn.ConvTranspose2d(16, 3, 11, stride=1, padding=1), # N, 1, 28, 28  (N,1,27,27)
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

    model = ConvolutionalAutoencoder(3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    num_epochs = 100
    outputs = []
    for epoch in range(num_epochs):
        i = 0
        for (data, _) in train_loader:
            # print(len(train_loader))
            # input()
            # Converts `data` to a tensor with a specified dtype
            # data = data.to(device)
            # Info on this https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch0
            # img = data.reshape(-1, 28*28) # -> use for Autoencoder_Linear
            # img = data
            i+=1
            recon = model(data)
            loss = criterion(recon,data)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(recon.shape)
            # input()
            save_image(recon.view(recon.shape[0], 3, 28, 28), f"./training/v2image{epoch+1}.png")
            print(f'{i}:Epoch:{epoch+1}, Loss:{loss.item():.4f}')
            outputs.append((epoch, data, recon))

    # for k in range(0, num_epochs):
    #     plt.figure(figsize=(9, 2))
    #     # plt.gray()
    #     imgs = outputs[k][1].detach().numpy()
    #     recon = outputs[k][2].detach().numpy()
    #     for i, item in enumerate(imgs):
    #         if i >= 9: break
    #         plt.subplot(2, 9, i+1)
    #         # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
    #         # item: 1, 28, 28
    #         plt.imshow(item[0])
                
    #     for i, item in enumerate(recon):
    #         if i >= 9: break
    #         plt.subplot(2, 9, 9+i+1) # row_length + i + 1
    #         # item = item.reshape(-1, 28,28) # -> use for Autoencoder_Linear
    #         # item: 1, 28, 28
    #         plt.imshow(item[0])
    #     plt.show()
    torch.save(model, "./CAE-100.pt")
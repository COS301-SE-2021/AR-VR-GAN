from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DataLoaders:
    def __init__(self, transform: transforms = None, kwargs: dict = {}, batch_size: int = 32):
        """This class is used to select specific datasets and adjust them as the user pleases
        """
        self.transform: transforms = transform
        self.batch_size: int = batch_size
        self.kwargs: dict = kwargs
        if transform == None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        
        self.datasets: dict = {
            "mnist": datasets.MNIST('./data', train=True, download=True, transform=self.transform), 
            "fashion": datasets.FashionMNIST('./data', train=True, download=True, transform=self.transform),
            "cifar10": datasets.CIFAR10('./data', train=True, transform=self.transform),
            "celeba": datasets.CelebA('./data', split="train", transform=self.transform)
        }
        
    def get_dataloader(self, name: str = "mnist") -> DataLoader:
        """This function is used to retrieve a dataloader specified in name. 
        If a name is not specified it returns the MNIST data and it returns the 
        MNIST if an invalid name is passed through."""
        if name in self.datasets:
            return DataLoader(self.datasets[name], batch_size=self.batch_size, shuffle=True, **self.kwargs)
        else:
            return DataLoader(self.datasets["mnist"], batch_size=self.batch_size, shuffle=True, **self.kwargs)
        
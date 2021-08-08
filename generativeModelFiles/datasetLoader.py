from torch.utils.data import DataLoader
from torch.utils.data import dataset
from torch import cuda
from torchvision import datasets, transforms
import pickle

# Cannot pickle train_loader objects as the size is too large 45 MB for the MNIST data loader 
# def save_object(obj, filename):
#     with open(filename, 'wb') as outp:  # Overwrites any existing file.
#         pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda.is_available() else {}

# train_loader = DataLoader(
#             datasets.MNIST('../data', train=True, download=True,
#                         transform=transforms.ToTensor()),
#             batch_size= 128, shuffle=True, **kwargs)

# test_loader = DataLoader(
#             datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#             batch_size=128, shuffle=True, **kwargs)

# save_object(train_loader, 'defaultDatasets/MNIST_128_train.pkl')
# save_object(train_loader, 'defaultDatasets/MNIST_128_test.pkl')

class DataLoaderHandler:

    def __init__(self) -> None:
        self.availableDataLoaders: dict = {}
        self.number_of_dataloaders: int = 0
        self.generateDefaults()

    def createDataLoader(self, dataset) -> None:
        pass

    def getDataLoader(self, key: str) -> DataLoader:
        value = self.availableDataLoaders.get(key)

        if value == None:
            # Raise an exception
            pass

        return value


    def generateDefaults(self) -> None:
        train_loader: DataLoader = None
        test_loader: DataLoader = None
        
        kwargs = {'num_workers': 1, 'pin_memory': True} if cuda.is_available() else {}

        # Saving MNIST data loader
        train_loader = DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.ToTensor()),
            batch_size= 128, shuffle=True, **kwargs)

        test_loader = DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
            batch_size=128, shuffle=True, **kwargs)

        loader_pair = [train_loader, test_loader]
        self.availableDataLoaders["MNIST"] = loader_pair

        # Does not work have to manually download the files

        # Saving CelebA data loader
        # train_loader = DataLoader(
        #     datasets.CelebA('../data', split="train", download=True,
        #                 transform=transforms.ToTensor()),
        #     batch_size= 128, shuffle=True, **kwargs)

        # test_loader = DataLoader(
        #     datasets.CelebA('../data', split="test", transform=transforms.ToTensor()),
        #     batch_size=128, shuffle=True, **kwargs)

        # loader_pair = [train_loader, test_loader]
        # self.availableDataLoaders["CelebA"] = loader_pair

        # Saving FashionMNIST data loader
        train_loader = DataLoader(
            datasets.FashionMNIST('../data', train=True, download=True,
                        transform=transforms.ToTensor()),
            batch_size= 128, shuffle=True, **kwargs)

        test_loader = DataLoader(
            datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor()),
            batch_size=128, shuffle=True, **kwargs)

        loader_pair = [train_loader, test_loader]
        self.availableDataLoaders["FashionMNIST"] = loader_pair

        # Saving EMNIST data loader
        # create an error handler for these error
        # OSError: [WinError 10051] A socket operation was attempted to an unreachable network
        # urllib.error.URLError: <urlopen error [WinError 10051] A socket operation was attempted to an unreachable network>  
        # train_loader = DataLoader(
        #     datasets.EMNIST('../data', split="letters", train=True, download=True,
        #                 transform=transforms.ToTensor()),
        #     batch_size= 128, shuffle=True, **kwargs)

        # test_loader = DataLoader(
        #     datasets.EMNIST('../data', split="letters", train=False, transform=transforms.ToTensor()),
        #     batch_size=128, shuffle=True, **kwargs)

        # loader_pair = [train_loader, test_loader]
        # self.availableDataLoaders["ENISTletters"] = loader_pair

        # zipfile.BadZipFile: File is not a zip file
        # Saving WIDERFace data loader
        # train_loader = DataLoader(
        #     datasets.WIDERFace('../data', split="train", download=True,
        #                 transform=transforms.ToTensor()),
        #     batch_size= 128, shuffle=True, **kwargs)

        # test_loader = DataLoader(
        #     datasets.WIDERFace('../data', train=False, transform=transforms.ToTensor()),
        #     batch_size=128, shuffle=True, **kwargs)

        # loader_pair = [train_loader, test_loader]
        # self.availableDataLoaders["WIDERFace"] = loader_pair

        # Saving Cityscapes data loader
        # train_loader = DataLoader(
        #     datasets.Cityscapes('../data', split="train", download=True,
        #                 transform=transforms.ToTensor()),
        #     batch_size= 128, shuffle=True, **kwargs)

        # test_loader = DataLoader(
        #     datasets.Cityscapes('../data', split="test", transform=transforms.ToTensor()),
        #     batch_size=128, shuffle=True, **kwargs)

        # loader_pair = [train_loader, test_loader]
        # self.availableDataLoaders["Cityscapes"] = loader_pair
        
dlh = DataLoaderHandler()
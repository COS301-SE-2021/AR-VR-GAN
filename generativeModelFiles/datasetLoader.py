from torch.utils.data import DataLoader
from torch.utils.data import dataset
from torch import cuda
from torchvision import datasets, transforms
import pickle

# Cannot pickle train_loader objects as the size is too large 45 MB for the MNIST data loader 
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda.is_available() else {}

train_loader = DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.ToTensor()),
            batch_size= 128, shuffle=True, **kwargs)

test_loader = DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
            batch_size=128, shuffle=True, **kwargs)

save_object(train_loader, 'defaultDatasets/MNIST_128_train.pkl')
save_object(train_loader, 'defaultDatasets/MNIST_128_test.pkl')

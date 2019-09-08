import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

use_cuda = torch.cuda.is_available()
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {'num_workers': 4}

data_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])


class MnistDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.x, self.y = data
        self.transform = transform

    def __getitem__(self, index):
        x = self.x
        if self.transform is not None:
            x = self.transform(x)
        self.x = x
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def data_loader(batch_size):
    x_train = np.load('data/x_train.npy')
    x_train = torch.tensor(x_train).squeeze(1)
    x_train = x_train.reshape(-1, 1, 28, 28).float()
    y_train = np.load('data/y_train.npy')
    y_train = torch.tensor(y_train).long()

    x_test = np.load('data/x_test.npy')
    x_test = torch.tensor(x_test).squeeze(1)
    x_test = x_test.reshape(-1, 1, 28, 28).float()
    y_test = np.load('data/y_test.npy')
    y_test = torch.tensor(y_test).long()

    train_data = x_train, y_train
    train_dataset = MnistDataset(train_data)

    val_data = x_test, y_test
    val_dataset = MnistDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader

import torch
import torchvision
from torchvision.transforms import Compose, Lambda, ToTensor


def load_mnist(train=True, shrinkage=None):
    dataset = torchvision.datasets.MNIST(
        root='.',
        download=True,
        train=train,
        transform=Compose([ToTensor(), Lambda(torch.flatten)])
    )
    if shrinkage:
        dataset_size = len(dataset)
        perm = torch.randperm(dataset_size)
        idx = perm[:int(dataset_size * shrinkage)]
        return torch.utils.data.Subset(dataset, idx)
    return dataset

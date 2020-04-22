import torch
import torchvision
from torch.utils import data
import torchvision.transforms as transforms

def train_loader(batch_size=64):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(root='/mnt/disk50/datasets/cifar',train=True,
                                             download=False, transform=transform_train)
    return  data.DataLoader(train_set, batch_size=batch_size,shuffle=True)

def test_loader(batch_size=64):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(root='/mnt/disk50/datasets/cifar', train=False,
                                           download=False, transform=transform_test)

    return torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)



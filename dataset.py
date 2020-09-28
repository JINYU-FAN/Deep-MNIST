import torch
from torchvision import datasets, transforms


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./dataset/', train=True, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=5, shuffle=True)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./dataset', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)


Fashion_train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./dataset/', train=True, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=5, shuffle=True)


Fashion_test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('./dataset', train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
        batch_size=1, shuffle=True)
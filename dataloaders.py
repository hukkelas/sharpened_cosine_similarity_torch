import torchvision
import os
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloader(batch_size: int):
    training_set = CIFAR10(
        root=os.path.join('.', 'data', 'cifar10'),
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))
    testing_set = CIFAR10(
        root=os.path.join('.', 'data', 'cifar10'),
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]))


    training_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        prefetch_factor=8,
        pin_memory=True,
        drop_last=True
        )
    testing_loader = DataLoader(
        testing_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=True
    )
    return training_loader, testing_loader

from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

import numpy as np 


def get_cifar100_train(batch_size = 10, val_split = 0.2):
    
    # Define a transformation
    transform =  transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding = 4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = (0.5071, 0.4867, 0.4408),
            std = (0.2675, 0.2565, 0.2761)
        )
    ])
    
    # Use this on the loaded CIFAR-100 data
    train_data = datasets.CIFAR100(
        root = "./data",
        train = True, 
        download = True, 
        transform = transform
    )
    
    # Define the sizes
    valid_size = int(len(train_data) * val_split)
    
    train_size = len(train_data) - valid_size
    
    # Random splitting
    train_dataset, valid_dataset =  random_split(
        train_data, [train_size, valid_size] 
    )
    
    # Now create Dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True, 
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size = batch_size,
        shuffle = False, 
    )
    
    return train_dataloader, valid_dataloader
    

def get_cifar100_test(batch_size = 10):
    
    # Define a transformation
    transform =  transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean = (0.5071, 0.4867, 0.4408),
            std = (0.2675, 0.2565, 0.2761)
        )
    ])
    
    # Use this on the loaded CIFAR-100 data
    data = datasets.CIFAR100(
        root = "./data",
        train = False, 
        download = True, 
        transform = transform
    )
    
    dataloader = DataLoader(
        data,
        batch_size = batch_size,
        shuffle = False, 
    )
    
    return dataloader
    
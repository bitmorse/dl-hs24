import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# Select certain classes from the Torchvision FashionMNIST dataset
def split(dataset, classes): 
    indices = []
    for i in range(len(dataset)):
        if dataset.targets[i] in classes:
            indices.append(i)
    return torch.utils.data.Subset(dataset, indices)

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

# Select a certain ratio of the dataset for each class
def select(dataset, ratio):
    subset_indices = []
    for target in dataset.targets.unique():
        indices = []
        for i in range(len(dataset)):
            if dataset.targets[i] == target:
                indices.append(i)
        np.random.shuffle(indices)
        indices = indices[:int(ratio*len(indices))]
        subset_indices.extend(indices)

    return torch.utils.data.Subset(dataset, subset_indices)
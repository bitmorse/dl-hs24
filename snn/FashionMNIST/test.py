from data import split
from model import SNN, train_snn, test_snn

from spikingjelly.activation_based import encoding, surrogate

import torch
from torchvision import datasets, transforms

batch_size = 64
dataset_name = 'FashionMNIST'
data_path=f'/archive/{dataset_name}'
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])
train_dt = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
test_dt = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)

train_p1, train_p2 = split(train_dt, [0, 1, 2, 3, 4]), split(train_dt, [5, 6, 7, 8, 9])
test_p1, test_p2 = split(test_dt, [0, 1, 2, 3, 4]), split(test_dt, [5, 6, 7, 8, 9])

train_loader_p1 = torch.utils.data.DataLoader(train_p1, batch_size=batch_size, shuffle=True)
train_loader_p2 = torch.utils.data.DataLoader(train_p2, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dt, batch_size=batch_size, shuffle=False)
test_p1_loader = torch.utils.data.DataLoader(test_p1, batch_size=batch_size, shuffle=False)
test_p2_loader = torch.utils.data.DataLoader(test_p2, batch_size=batch_size, shuffle=False)

net = SNN(surrogate_function=surrogate.Sigmoid(), threshold=1.0, num_steps=10).to("cuda")
encoder = encoding.PoissonEncoder()

optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
criterion = torch.nn.CrossEntropyLoss()

train_snn(net, train_loader_p1, criterion, optimizer, encoder, 1)
print("Testing model with first half of the dataset")
test_snn(net, test_p1_loader, encoder)
print("Testing model with second half of the dataset")
test_snn(net, test_p2_loader, encoder)
print("Testing model with all of the dataset")
test_snn(net, test_loader, encoder)

train_snn(net, train_loader_p2, criterion, optimizer, encoder, 1)
print("Testing model with first half of the dataset")
test_snn(net, test_p1_loader, encoder)
print("Testing model with second half of the dataset")
test_snn(net, test_p2_loader, encoder)
print("Testing model with all of the dataset")
test_snn(net, test_loader, encoder)

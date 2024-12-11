from data import split
from ga import GeneticAlgorithmNN
from model import ANN, train_ann, test_ann

import torch
from torchvision import datasets, transforms

batch_size = 64
dataset_name = 'FashionMNIST'
data_path=f'/scratch/zyi/codeSpace/data/{dataset_name}'
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])
train_dt = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
test_dt = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)

train_p1, train_p2 = split(train_dt, [0, 1, 2, 3, 4]), split(train_dt, [5, 6, 7, 8, 9])

train_loader_p1 = torch.utils.data.DataLoader(train_p1, batch_size=batch_size, shuffle=True)
train_loader_p2 = torch.utils.data.DataLoader(train_p2, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dt, batch_size=batch_size, shuffle=False)

net1 = ANN()
net2 = ANN()

optimizer1 = torch.optim.Adam(net1.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(net2.parameters(), lr=0.001)

criterion = torch.nn.CrossEntropyLoss()
train_ann(net1, train_loader_p1, criterion, optimizer1, 1)
train_ann(net2, train_loader_p2, criterion, optimizer2, 1)

print("Testing model 1")
test_ann(net1, test_loader)
print("Testing model 2")
test_ann(net2, test_loader)

genAlg = GeneticAlgorithmNN([net1, net2], mutation_rate=0.1, crossover_rate=0.5)
for i in range(4):
    genAlg.add_mutants(net1)
    genAlg.add_mutants(net2)

num_generations = 2
genAlg.evolve(test_loader, num_generations)

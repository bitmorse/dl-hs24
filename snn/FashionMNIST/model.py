import torch
import torch.nn as nn
import numpy as np

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import functional

class SNN(nn.Module):
    def __init__(self, surrogate_function, threshold=1.0, num_steps=10):
        super(SNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.mp1 = nn.MaxPool2d(2)
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate_function, v_threshold=threshold)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.mp2 = nn.MaxPool2d(2)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate_function, v_threshold=threshold)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*4*4, 10)
        self.lif3 = neuron.LIFNode(surrogate_function=surrogate_function, v_threshold=threshold)

        self.num_steps = num_steps
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.lif1(x)

        x = self.conv2(x)
        x = self.mp2(x)
        x = self.lif2(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.lif3(x)

        return x
    
def forward_pass(net, data, encoder):
    out_fr = 0.
    img = encoder(data).to("cuda")
    for step in range(net.num_steps):
        out_fr += net(img)
    return out_fr

def train_snn(model, train_loader, criterion, optimizer, encoder, num_epochs):
    accs = []

    model.train()
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(iter(train_loader)):
            images, labels = images.to("cuda"), labels.to("cuda")

            functional.reset_net(model)

            out_fr = forward_pass(model, images, encoder)
            loss_val = criterion(out_fr, labels)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            _, idx = out_fr.max(1)
            acc = np.mean((labels == idx).detach().cpu().numpy())

            accs.append(acc)

            if i % 25 == 0:
                print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
                print(f"Accuracy: {acc * 100:.2f}%\n")

    return accs

def test_snn(model, test_loader, encoder):
    model.eval()
    
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(iter(test_loader)):
            images, labels = images.to("cuda"), labels.to("cuda")
            functional.reset_net(model)
            out_fr = forward_pass(model, images, encoder)
            _, idx = out_fr.max(1)
            correct += (labels == idx).sum().item()
            total += labels.size(0)
    
    acc = correct / total
    print(f"Test Accuracy: {acc * 100:.2f}%\n")

    return acc

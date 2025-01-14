import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import lightning as L

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import encoding, surrogate

class SNN(nn.Module):
    def __init__(self, surrogate_function, threshold=0.2, num_steps=64):
        super(SNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.mp1 = nn.MaxPool2d(2)
        self.lif1 = neuron.LIFNode(surrogate_function=surrogate_function, v_threshold=threshold, tau=1.2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.mp2 = nn.MaxPool2d(2)
        self.lif2 = neuron.LIFNode(surrogate_function=surrogate_function, v_threshold=threshold, tau=1.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*4*4, 10)
        #self.lif3 = neuron.LIFNode(surrogate_function=surrogate_function, v_threshold=threshold, tau=1.2)

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
        #x = self.lif3(x)

        return x


class MultiStepSNN(SNN):
    def __init__(self):
        super(MultiStepSNN, self).__init__(surrogate_function=surrogate.PiecewiseLeakyReLU(w=2,c=0.01))

    def forward(self, x: torch.Tensor):
        out_fr = 0.
        functional.reset_net(self)
        encoder = encoding.PoissonEncoder()
        img = encoder(x).to("cuda")
        for step in range(self.num_steps):
            out_fr += super(MultiStepSNN, self).forward(img)
        return out_fr


class LightningSNN(L.LightningModule):
    def __init__(self, learning_rate=0.001):
        super(LightningSNN, self).__init__()
        self.model = SNN(surrogate_function= surrogate.PiecewiseLeakyReLU(w=2,c=0.01))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.correct = 0
        self.total = 0
        self.encoder = encoding.PoissonEncoder()
        
    def training_step(self, batch, batch_idx):
        # Unpack batch
        images, labels = batch
        images = images.to('cuda')
        labels = labels.to('cuda')
        functional.reset_net(self.model)

        # Forward pass
        out_fr = 0.
        img = self.encoder(images).to("cuda")
        for _ in range(self.model.num_steps):
            out_fr += self.model(img)

        loss = self.criterion(out_fr, labels)

        return loss
    
    def on_test_start(self):
        self.correct = 0
        self.total = 0
        
    def test_step(self, batch, batch_idx):
        images, labels = batch
        
        images = images.to('cuda')
        labels = labels.to('cuda')
        functional.reset_net(self.model)

        # Forward pass
        out_fr = 0.
        img = self.encoder(images).to("cuda")
        for _ in range(self.model.num_steps):
            out_fr += self.model(img)

        _, idx = out_fr.max(1)
        self.correct += (labels == idx).sum().item()
        self.total += labels.size(0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    

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

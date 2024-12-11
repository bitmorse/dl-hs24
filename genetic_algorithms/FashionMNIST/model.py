import torch
import torch.nn as nn
import numpy as np

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.mp1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.mp2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*4*4, 10)

        self.acc_threshold = 0.8
        self.reached_threshold = False
        self.steps_to_threshold = 0

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.mp2(x)
        x = self.relu2(x)

        x = self.flatten(x)
        x = self.fc1(x)

        return x

def train_ann(model, train_loader, criterion, optimizer, num_epochs):
    accs = []

    model.to("cuda")
    model.train()
    log_softmax = nn.LogSoftmax(dim=-1)
    
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(iter(train_loader)):
            images, labels = images.to("cuda"), labels.to("cuda")

            optimizer.zero_grad()
            out = model(images)
            loss_val = criterion(log_softmax(out), labels)
            loss_val.backward()
            optimizer.step()

            _, idx = out.max(1)
            acc = np.mean((labels == idx).detach().cpu().numpy())

            accs.append(acc)

            if i % 25 == 0:
                print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
                print(f"Accuracy: {acc * 100:.2f}%\n")

    model.to("cpu")

    return accs

def test_ann(model, test_loader):
    model.to("cuda")
    model.eval()
    log_softmax = nn.LogSoftmax(dim=-1)

    total = 0
    correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(iter(test_loader)):
            images, labels = images.to("cuda"), labels.to("cuda")
            out = model(images)
            _, idx = out.max(1)
            correct += (labels == idx).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Test Accuracy: {acc * 100:.2f}%\n")

    model.to("cpu")

    return acc

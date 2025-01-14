import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class ANN(nn.Module):
    def __init__(self, state_dict=None):
        super(ANN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.mp1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.mp2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*4*4, 10)
        
        self.origin = "new"
        if state_dict is not None:
            self.load_state_dict(state_dict)
      
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

def compute_weight_importance(model, train_loader, criterion):
    """
    returns dict where keys are parameter names and values are importance scores.
    """
    model.to("cuda")
    model.eval()
    importance = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    for images, labels in train_loader:
        images, labels = images.to("cuda"), labels.to("cuda")
        model.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward() 

        #accumulate gradient norms
        for name, param in model.named_parameters():
            if param.grad is not None:
                importance[name] += param.grad.abs()

    #normalize
    for name in importance:
        importance[name] /= len(train_loader.dataset)

    model.to("cpu")
    return importance

def train_ann(model, train_loader, criterion, optimizer, num_epochs, gpu2cpu=True, slurm=False):
    accs = []
    if gpu2cpu:
        model.to("cuda")
    model.train()
    log_softmax = nn.LogSoftmax(dim=-1)

    for epoch in range(num_epochs):
        if not slurm:
            progress_bar = tqdm(
                enumerate(train_loader), 
                total=len(train_loader), 
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                unit="batch"
            )
        
            for i, (images, labels) in progress_bar:
                images, labels = images.to("cuda"), labels.to("cuda")

                optimizer.zero_grad()
                out = model(images)
                loss_val = criterion(log_softmax(out), labels)
                loss_val.backward()
                optimizer.step()

                _, idx = out.max(1)
                acc = np.mean((labels == idx).detach().cpu().numpy())
                accs.append(acc)

                # Update progress bar description
                progress_bar.set_postfix({"Loss": f"{loss_val.item():.2f}", "Accuracy": f"{acc * 100:.2f}%"})
        else:
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to("cuda"), labels.to("cuda")

                optimizer.zero_grad()
                out = model(images)
                loss_val = criterion(log_softmax(out), labels)
                loss_val.backward()
                optimizer.step()

                _, idx = out.max(1)
                acc = np.mean((labels == idx).detach().cpu().numpy())
                accs.append(acc)

    if gpu2cpu:
        model.to("cpu")

    return accs

def test_ann(model, test_loader, gpu2cpu=True):
    if gpu2cpu:
        model.to("cuda")
    model.eval()

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
    #print(f"Test Accuracy: {acc * 100:.2f}%\n")

    if gpu2cpu:
        model.to("cpu")

    return acc

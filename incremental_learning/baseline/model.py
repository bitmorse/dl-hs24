import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
import torchvision

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


class LightningANN(L.LightningModule):
    def __init__(self, learning_rate=0.001, model_type:callable=ANN):
        super(LightningANN, self).__init__()
        self.model = model_type()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.correct = 0
        self.total = 0
        
    def training_step(self, batch, batch_idx):
        # Unpack batch
        images, labels = batch

        # Forward pass
        out = self.model(images)
        #loss = self.criterion(F.log_softmax(out, dim=-1), labels)
        loss = self.criterion(out, labels)

        return loss
    
    def on_test_start(self):
        self.correct = 0
        self.total = 0
        
    def test_step(self, batch, batch_idx):
        images, labels = batch
        out = self.model(images)
        
        _, idx = out.max(1)
        self.correct += (labels == idx).sum().item()
        self.total += labels.size(0)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
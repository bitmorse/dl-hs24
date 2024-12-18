from torchvision import datasets, transforms
from incremental_trainer import IncrementalTrainer
from interfaces import TrainingSessionInterface
from baseline.model import LightningANN

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L


class BaselineTrainingSession(TrainingSessionInterface):
    def __init__(self, hyperparams: dict):
        self.hyperparams = hyperparams
        self.model = LightningANN(learning_rate=hyperparams['lr'])
        self.trainer_params = {
            'max_epochs':hyperparams['num_epochs'], 
            'enable_checkpointing':False
        }
        self.trainer = L.Trainer(**self.trainer_params)
    
    def init_model(self, full_file_path: str):
        self.model = LightningANN.load_from_checkpoint(full_file_path)
        self.trainer = L.Trainer(**self.trainer_params) #reset the trainer
        
    def fit(self, train_dt, base_replay_dt=None):
        if base_replay_dt is not None:
            train_dt = torch.utils.data.ConcatDataset([train_dt, base_replay_dt])
   
        train_loader = DataLoader(train_dt, batch_size=self.hyperparams['batch_size'], shuffle=True)

        self.trainer.fit(self.model, train_loader)
        
    
    def test(self, test_dt):
        self.model.eval()
        test_loader = DataLoader(test_dt, batch_size=self.hyperparams['batch_size'], shuffle=False)
        result = self.trainer.test(self.model, test_loader)
        test_acc = self.model.correct / self.model.total
        return test_acc
    
    def save_model(self, full_file_path: str):
        print("save model")
        self.trainer.save_checkpoint(full_file_path, weights_only=True)
        pass
    

def main():
    batch_size = 64
    dataset_name = 'FashionMNIST'
    data_path=f'/tmp/{dataset_name}'
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    train_dt = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
    test_dt = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
    
    hyperparameters_session = {
        'batch_size': 64,
        'num_epochs': 1,
        'lr': 0.001,
        'train_val_ratio': 0.8
    }
    
    incremental_trainer_config = {
        'replay_buffer_size': 100,
        'training_sessions': 6,
        'base_classes': [0,1,2,3,4],
        'incremental_classes_total': [5,6,7,8,9],
        'incremental_classes_per_session': 1
    }
    
    session_trainer = BaselineTrainingSession(hyperparameters_session) #exchange with your own session trainer
    trainer = IncrementalTrainer(session_trainer, train_dt, test_dt, 
                                 "/tmp/checkpoints", incremental_trainer_config)
    trainer.train()
    trainer.save_metrics()
    
if __name__ == "__main__":
    main()
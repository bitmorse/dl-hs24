from torchvision import datasets, transforms
from incremental_trainer import IncrementalTrainer
from interfaces import TrainingSessionInterface
from baseline.model import LightningANN
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from genetic_algorithms.FashionMNIST.ga import GeneticAlgorithmNN
from genetic_algorithms.FashionMNIST.model import ANN, train_ann, test_ann
import pickle

class GATrainingSession(TrainingSessionInterface):
    def __init__(self, hyperparams: dict):
        self.hyperparams = hyperparams
        self.model = None
        self.criterion = torch.nn.CrossEntropyLoss()

    def init_model(self, full_file_path: str): 
        return
        #load pickle self.model from file 
        with open(full_file_path, 'rb') as f:
            self.model = pickle.load(f)
            
        
    def fit(self, train_dt, base_replay_dt=None):
        train_loader = DataLoader(train_dt, batch_size=self.hyperparams['batch_size'], shuffle=True)

        #if initial model is None, train a new model A and consider it base model
        if self.model is None:
            print("Train a base model A. No evolution is performed.")
            
            netA = ANN()
            optimizerA = torch.optim.Adam(netA.parameters(), lr=self.hyperparams['lr'])
            train_ann(netA, train_loader, self.criterion, optimizerA, 1)
            
            self.model = netA

        else:
            print("Load model A. Train a new model B (w/ incremental class data). \
                  Then uses B and A in GA initial population. evolution outputs best model C and saves it.")
            
            train_and_replay_dt = torch.utils.data.ConcatDataset([train_dt, base_replay_dt])
            train_and_replay_loader = DataLoader(train_and_replay_dt, batch_size=self.hyperparams['batch_size'], shuffle=True)

            netA = self.model
            netB = ANN()
            optimizerB = torch.optim.Adam(netB.parameters(), lr=self.hyperparams['lr'])
            train_ann(netB, train_loader, self.criterion, optimizerB, 1)
            mr = self.hyperparams['mutation_rate']
            cr = self.hyperparams['crossover_rate']
            sr = self.hyperparams['selection_ratio']
            ga = GeneticAlgorithmNN([netA, netB], mutation_rate=mr, crossover_rate=cr)

            for i in range(self.hyperparams['initial_population_size']//2):
                ga.add_mutants(netA)
                ga.add_mutants(netB)
                
            self.model = ga.evolve(train_and_replay_loader, train_loader, self.hyperparams['num_generations'], selection_ratio=sr)
        
    
    def test(self, test_dt):
        test_loader = DataLoader(test_dt, batch_size=self.hyperparams['batch_size'], shuffle=False)
        accuracy = test_ann(self.model, test_loader)
        return accuracy
    
    def save_model(self, full_file_path: str):
        return
        #create path if not exists
        if not os.path.exists(os.path.dirname(full_file_path)):
            os.makedirs(os.path.dirname(full_file_path))
        #pickle self.model to file 
        with open(full_file_path, 'wb') as f:
            pickle.dump(self.model, f)
        
    

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
        'train_val_ratio': 0.8,
        'mutation_rate': 0.8,
        'crossover_rate': 0.5,
        'selection_ratio': [0.4, 0.3, 0.1, 0.2],#children,mutants,elites,new
        'num_generations': 10,
        'initial_population_size': 100
    }
    
    incremental_trainer_config = {
        'replay_buffer_size': 5000,
        'training_sessions': 6,
        'base_classes': [0,1,2,3,4],
        'incremental_classes_total': [5,6,7,8,9],
        'incremental_classes_per_session': 1
    }
    
    baseline_session = BaselineTrainingSession(hyperparameters_session) #exchange with your own session trainer
    ga_session = GATrainingSession(hyperparameters_session) #exchange with your own session trainer
    
    #train GA session
    trainer1 = IncrementalTrainer(ga_session, train_dt, test_dt, 
                                 "/tmp/checkpoints", incremental_trainer_config)
    trainer1.train()
    trainer1.save_metrics()
    
    #train baseline session
    trainer2 = IncrementalTrainer(baseline_session, train_dt, test_dt,
                                    "/tmp/checkpoints", incremental_trainer_config)
    trainer2.train()
    trainer2.save_metrics()
    
    
if __name__ == "__main__":
    main()
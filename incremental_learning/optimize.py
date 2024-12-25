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
import optuna
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from genetic_algorithms.FashionMNIST.ga import GeneticAlgorithmNN
from genetic_algorithms.FashionMNIST.model import ANN, train_ann, test_ann
import pickle

from sessions import GATrainingSession, BaselineTrainingSession

dataset_name = 'FashionMNIST'
data_path=f'/tmp/{dataset_name}'
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])
train_dt = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
test_dt = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)

def hyperparam_objective(trial):
    
    hyperparameters_session = {
        'batch_size': 64,
        'num_epochs': 1,
        'lr': 0.001,
        'mutation_scale': trial.suggest_float('crossover_rate', 0.1, 0.5, step=0.2),

        'mutation_rate': trial.suggest_float('mutation_rate', 0.1, 0.7, step=0.2),
        'crossover_rate': trial.suggest_float('crossover_rate', 0.1, 0.5, step=0.2),
        'selection_ratio': [0.5, 0.2, 0.2, 0.1],#children,mutants,elites,new
        'num_generations': trial.suggest_int('num_generations', 40, 200, step=20),
        'initial_population_size': trial.suggest_int('initial_population_size', 10, 30, step=10),
        'recall_importance':  trial.suggest_float('recall_importance', 0.4, 0.7, step=0.1),
        'parent_selection_strategy': trial.suggest_categorical('parent_selection_strategy', ['combined', 'pareto']),
        'selection_ratio': [0.5, 0.2, 0.2, 0.1],#children,mutants,elites,new
        'crossover_strategy': trial.suggest_categorical('crossover_strategy', ['none', 'random', 'importance'])
        #none, random, importance
        
    }
    
    incremental_trainer_config = {
        'replay_buffer_size':   trial.suggest_int('replay_buffer_size', 1000, 5000, step=1000),
        'incremental_training_size': 300,

        'training_sessions': 6,
        'base_classes': [0,1,2,3,4],
        'incremental_classes_total': [5,6,7,8,9],
        'incremental_classes_per_session': 1
    }
    
    baseline_session = BaselineTrainingSession(hyperparameters_session) #exchange with your own session trainer
    ga_session = GATrainingSession(hyperparameters_session) #exchange with your own session trainer
    
    #train baseline session
    trainer2 = IncrementalTrainer(ga_session, train_dt, test_dt,
                                    "/tmp/checkpoints", incremental_trainer_config)
    trainer2.train()
    objective = trainer2.get_cf_metric('omega_all')
    
    return -objective

def opt_process():
    #hyperparameter optimization        
    study = optuna.load_study(storage="sqlite:///db.sqlite3", study_name="ga2")
    study.optimize(hyperparam_objective, n_trials=100)

    print("Best hyperparameters:")
    print(study.best_params)
    
def main():
    #start n processes each 1 core of cpu
    import multiprocessing
    processes = []
    for i in range(25):
        p = multiprocessing.Process(target=opt_process)
        p.start()
        processes.append(p)

    
if __name__ == "__main__":
    main()
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
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from genetic_algorithms.ga import GeneticAlgorithmNN
from genetic_algorithms.model import ANN, train_ann, test_ann
from snn.FashionMNIST.model import MultiStepSNN
import pickle

from sessions import GATrainingSession, BaselineTrainingSession, PyGADTrainingSession

from config import get_datasets, INCREMENTAL_TRAINER_CONFIG

def run_experiment(experiment_id):
    train_dt, test_dt = get_datasets(data_path='/tmp')

    hyperparameters_session = {
        'model_type': ANN,
        'transfer': False,
        'baseline_model_type': ANN,
        'batch_size': 64,
        'num_epochs': 1,
        'lr': 0.001,
        'num_generations': 200,
        'num_parents_mating': 5,
        'population_size': 500,
        'parent_selection_type': "sss",
        'keep_parents': -1,
        'K_tournament': 3,
        'crossover_type': "single_point",
        'mutation_type': "random",
        'mutation_percent_genes': 10.0,
        'mutation_by_replacement': False,
        'random_mutation_min_val': -0.1,
        'random_mutation_max_val': 0.1,
        'fitness_batch_size': 500,
        'slurm': True
    }
    
    INCREMENTAL_TRAINER_CONFIG['enable_progress_bar'] = not hyperparameters_session['slurm']
    

    baseline_session = BaselineTrainingSession(hyperparameters_session) #exchange with your own session trainer
    ga_session = PyGADTrainingSession(hyperparameters_session) #exchange with your own session trainer

    #train baseline session
    trainer2 = IncrementalTrainer(baseline_session, train_dt, test_dt,
                                    "/tmp/checkpoints", INCREMENTAL_TRAINER_CONFIG, 
                                    experiment_id, alpha_ideal=None)
    trainer2.train()
    baseline_alpha_ideal = trainer2.get_alpha_ideal()
    if baseline_alpha_ideal is None:
        raise ValueError("Baseline alpha_ideal is None")
    trainer2.save_metrics()
    
    #train pygad session
    trainer1 = IncrementalTrainer(ga_session, train_dt, test_dt, 
                                 "/tmp/checkpoints", INCREMENTAL_TRAINER_CONFIG,
                                 experiment_id, alpha_ideal=baseline_alpha_ideal)
    trainer1.train()
    trainer1.save_metrics()

    
    # summarize cf metrics
    print("Baseline vs PyGAD session metrics")
    print(f"Omega All [baseline,pygad]: {trainer2.get_cf_metric('omega_all')}, {trainer1.get_cf_metric('omega_all')}")
    print(f"Omega Base [baseline,pygad]: {trainer2.get_cf_metric('omega_base')}, {trainer1.get_cf_metric('omega_base')}")
    print(f"Omega New [baseline,pygad]: {trainer2.get_cf_metric('omega_new')}, {trainer1.get_cf_metric('omega_new')}")
    
    #baseline session metrics
    #INFO:root:Omega Base: 0.8416872224963
    #INFO:root:Omega New: 0.9972
    #INFO:root:Omega All: 0.7397220851833579

if __name__ == "__main__":
    
    N_EXPERIMENTS = 2 #number of experiments to run for statistical significance
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    for i in range(N_EXPERIMENTS):
        experiment_id = f"experiment_pygad_{i}_{timestamp}"
        run_experiment(experiment_id)

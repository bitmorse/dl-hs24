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
from torch.utils.data import DataLoader, Dataset
import lightning as L
import time
import torchvision
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from genetic_algorithms.ga import GeneticAlgorithmNN
from genetic_algorithms.model import ANN, MLP, train_ann, test_ann, embeddingMLP
from snn.FashionMNIST.model import MultiStepSNN
import pickle

from sessions import GATrainingSession, BaselineTrainingSession, PyGADTrainingSession

from config import INCREMENTAL_TRAINER_CONFIG

class CIFAR10EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        self.targets = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def generate_cifar10_embeddings(batch_size=64):
    # Load the CIFAR-10 dataset
    dataset_name = 'CIFAR10'
    data_path=f'/tmp/{dataset_name}'
    # train_dt = datasets.CIFAR10(data_path, train=True, download=True, transform=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms())
    # test_dt = datasets.CIFAR10(data_path, train=False, download=True, transform=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms())
    train_dt = datasets.CIFAR10(data_path, train=True, download=True, transform=torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms())
    test_dt = datasets.CIFAR10(data_path, train=False, download=True, transform=torchvision.models.ResNet50_Weights.IMAGENET1K_V2.transforms())

    # Create a dataloader
    train_dataloader = DataLoader(train_dt, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dt, batch_size=batch_size, shuffle=False)

    # Load the pretrained ResNet18 model
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()

    # Remove the final classification layer (fc)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    # Use GPU if available
    device = torch.device("cuda")
    feature_extractor.to(device)

    # Extract embeddings for the training set
    train_embeddings = []
    train_labels = []

    with torch.no_grad():
        for images, targets in train_dataloader:
            images = images.to(device)

            # Extract features
            features = feature_extractor(images)
            features = features.view(features.size(0), -1)  # Flatten the output

            train_embeddings.append(features.cpu().numpy())
            train_labels.append(targets.numpy())

    # Concatenate all embeddings and labels
    train_embeddings = np.concatenate(train_embeddings, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    # Create a dataset with embeddings and labels
    train_embed_set = CIFAR10EmbeddingsDataset(train_embeddings, train_labels)

    # Extract embeddings for the test set
    test_embeddings = []
    test_labels = []

    with torch.no_grad():
        for images, targets in test_dataloader:
            images = images.to(device)

            # Extract features
            features = feature_extractor(images)
            features = features.view(features.size(0), -1)

            test_embeddings.append(features.cpu().numpy())
            test_labels.append(targets.numpy())

    # Concatenate all embeddings and labels
    test_embeddings = np.concatenate(test_embeddings, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    # Create a dataset with embeddings and labels
    test_embed_set = CIFAR10EmbeddingsDataset(test_embeddings, test_labels)

    return train_embed_set, test_embed_set

def run_experiment(experiment_id):

    train_dt, test_dt = generate_cifar10_embeddings()

    hyperparameters_session = {
        'model_type': embeddingMLP,
        'transfer': False,
        'baseline_model_type': embeddingMLP,
        'batch_size': 64,
        'num_epochs': 1,
        'lr': 0.001,
        'num_generations': 100,
        'num_parents_mating': 5,
        'population_size': 200,
        'parent_selection_type': "sss",
        'keep_parents': -1,
        'K_tournament': 3,
        'crossover_type': "single_point",
        'mutation_type': "random",
        'mutation_percent_genes': 10.0,
        'mutation_by_replacement': False,
        'random_mutation_min_val': -0.1,
        'random_mutation_max_val': 0.1,
        'fitness_batch_size': 200,
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
    print("Baseline vs PyGAD_MLP session metrics")
    print(f"Omega All [baseline,ga]: {trainer2.get_cf_metric('omega_all')}, {trainer1.get_cf_metric('omega_all')}")
    print(f"Omega Base [baseline,ga]: {trainer2.get_cf_metric('omega_base')}, {trainer1.get_cf_metric('omega_base')}")
    print(f"Omega New [baseline,ga]: {trainer2.get_cf_metric('omega_new')}, {trainer1.get_cf_metric('omega_new')}")

    # baseline session metrics
    # INFO:root:Omega Base: 0.8416872224963
    # INFO:root:Omega New: 0.9972
    # INFO:root:Omega All: 0.7397220851833579


if __name__ == "__main__":
    
    N_EXPERIMENTS = 1 #number of experiments to run for statistical significance
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    for i in range(N_EXPERIMENTS):
        experiment_id = f"experiment_pygad_embeddings_mlp_{i}_{timestamp}"
        run_experiment(experiment_id)

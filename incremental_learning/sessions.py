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
from genetic_algorithms.FashionMNIST.model import ANN, train_ann, test_ann, compute_weight_importance
from genetic_algorithms.FashionMNIST.pygad_interface import PyGADNN, init_population, fitness_func, on_generation
import pickle

class GATrainingSession(TrainingSessionInterface):
    def __init__(self, hyperparams: dict):
        self.hyperparams = hyperparams
        self.model = None
        self.base_model = None
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
            self.base_model = netA
        else:
            print("Load model A. Train a new model B (w/ incremental class data). \
                  Then uses B and A in GA initial population. evolution outputs best model C and saves it.")
            
            replay_loader = DataLoader(base_replay_dt, batch_size=self.hyperparams['batch_size'], shuffle=True)

            netA = self.model
            netB = ANN()
            optimizerB = torch.optim.Adam(netB.parameters(), lr=self.hyperparams['lr'])
            train_ann(netB, train_loader, self.criterion, optimizerB, 1)
            
            importance_netA = compute_weight_importance(netA, replay_loader, self.criterion)
            importance_netB = compute_weight_importance(netB, train_loader, self.criterion)
            
            ga = GeneticAlgorithmNN([netA, netB], 
                                    [importance_netA, importance_netB], 
                                    mutation_rate=self.hyperparams['mutation_rate'], 
                                    mutation_scale=self.hyperparams['mutation_scale'], 
                                    crossover_rate= self.hyperparams['crossover_rate'], 
                                    crossover_strategy=self.hyperparams['crossover_strategy'],
                                    model_args=[self.base_model.state_dict()],
                                    )
            
            self.model = ga.evolve(train_loader, replay_loader, self.hyperparams['num_generations'], 
                                   selection_ratio=self.hyperparams['selection_ratio'],
                                   recall_importance=self.hyperparams['recall_importance'], 
                                   parent_selection_strategy=self.hyperparams['parent_selection_strategy'],
                                   initial_population_size= self.hyperparams['initial_population_size']
                                   )
        
    
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
        
class PyGADTrainingSession(TrainingSessionInterface):
    def __init__(self, hyperparams: dict):
        self.hyperparams = hyperparams
        self.model = None
        self.base_model = None
        self.criterion = torch.nn.CrossEntropyLoss()

    def init_model(self, full_file_path: str): 
        pass

    def fit(self, train_dt, base_replay_dt=None):
        train_loader = DataLoader(train_dt, batch_size=self.hyperparams['batch_size'], shuffle=True)

        #if initial model is None, train a new model A and consider it base model
        if self.model is None:
            print("Train a base model A. No evolution is performed.")
            
            netA = ANN().to("cuda")
            optimizerA = torch.optim.Adam(netA.parameters(), lr=self.hyperparams['lr'])
            train_ann(netA, train_loader, self.criterion, optimizerA, 1, gpu2cpu=False)
            
            self.model = netA
            self.base_model = netA
        else:
            print("Load model A. Train a new model B (w/ incremental class data). \
                  Then uses B and A in GA initial population. evolution outputs best model C and saves it.")

            netA = self.model.to("cuda")
            netB = ANN().to("cuda")
            optimizerB = torch.optim.Adam(netB.parameters(), lr=self.hyperparams['lr'])
            train_ann(netB, train_loader, self.criterion, optimizerB, 1, gpu2cpu=False)

            merged_set = torch.utils.data.ConcatDataset([train_dt, base_replay_dt])
            merged_loader = DataLoader(merged_set, batch_size=len(merged_set), shuffle=False)

            ga = PyGADNN(
                model=ANN,
                train_loader=merged_loader,
                num_generations=self.hyperparams['num_generations'],
                num_parents_mating=self.hyperparams['num_parents_mating'],
                initial_population=init_population(self.hyperparams['population_size'], [netA, netB]),
                sol_per_pop=self.hyperparams['population_size'],
                parent_selection_type=self.hyperparams['parent_selection_type'],
                keep_parents=self.hyperparams['keep_parents'],
                K_tournament=self.hyperparams['K_tournament'],
                crossover_type=self.hyperparams['crossover_type'],
                mutation_type=self.hyperparams['mutation_type'],
                mutation_percent_genes=self.hyperparams['mutation_percent_genes'],
                mutation_by_replacement=self.hyperparams['mutation_by_replacement'],
                random_mutation_min_val=self.hyperparams['random_mutation_min_val'],
                random_mutation_max_val=self.hyperparams['random_mutation_max_val'],
                fitness_func=fitness_func,
                on_generation=on_generation
            )
            ga.run()

    def test(self, test_dt):
        test_loader = DataLoader(test_dt, batch_size=self.hyperparams['batch_size'], shuffle=False)
        accuracy = test_ann(self.model, test_loader, gpu2cpu=False)
        return accuracy

    def save_model(self, full_file_path: str):
        pass            

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
    
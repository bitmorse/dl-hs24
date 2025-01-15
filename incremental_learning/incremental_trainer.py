import os
import torch
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset
import lightning as L
import numpy as np
from interfaces import TrainingSessionInterface
import logging
import sys
import json
import time
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class IncrementalTrainer:
    def __init__(self, session_trainer: TrainingSessionInterface, 
                full_train_dt: Dataset, full_test_dt: Dataset, 
                checkpoints_path: str, config: dict, experiment_dir=None,
                alpha_ideal=None) -> None:
        self.session_trainer = session_trainer
        self.checkpoints_path = checkpoints_path
        self.experiment_name = f"{session_trainer.__class__.__name__}_{full_train_dt.__class__.__name__}"
        self.full_train_dt = full_train_dt
        self.full_test_dt = full_test_dt
        self.experiment_dir = experiment_dir
        
        if experiment_dir == None:
            self.experiment_dir = int(time.time())
            
            
        self.config = config

        # there a T training sessions. For each session i, we collect the following metrics.
        self.test_metrics = { 
            'alpha_base_sessions': [], # test accuracy on base set after i learning sessions
            'alpha_new_sessions': [], # test accuracy of session i after it is learned
            'alpha_all_sessions': [], # test accuracy for all test data and all classes seen to this point
            'alpha_ideal': 0, # test accuracy on the base set after first training. used for normalization
            'session_classes': [], # classes used in each session
        }
        # after all T training sessions, we compute the following metrics
        self.cf_metrics = {
            'omega_base': 0, # measure retention of the first session
            'omega_new': 0, # measure ability to recall the newly learned session
            'omega_all': 0, # measure retention of prior info AND aquisition of new info
        }
        
        if alpha_ideal == None:
            self.alpha_ideal = 0
            self._compute_alpha_ideal()
        else:
            logging.info("Using provided alpha_ideal.")
            self.alpha_ideal = alpha_ideal

        logging.info(f"Experiment name: {self.experiment_name}")
        logging.info("Dataset info:")
        logging.info(f"Full Train: {len(full_train_dt)} samples")
        logging.info(f"Full Test: {len(full_test_dt)} samples")
    
    def get_alpha_ideal(self):
        return self.alpha_ideal
    
    def get_cf_metric(self, metric_name):
        return self.cf_metrics.get(metric_name, None)
    
    def _compute_alpha_ideal(self):
        logging.info("Computing alpha_ideal (base set accuracy on 'offline model')...")
        ideal_trainer = self.session_trainer.__class__(self.session_trainer.hyperparams)
        ideal_trainer.fit(self.full_train_dt)
        base_test_dt = self._split(self.full_test_dt, self.config['base_classes'])
        self.test_metrics['alpha_ideal'] = ideal_trainer.test(base_test_dt)
        logging.info(f"alpha_ideal: {self.test_metrics['alpha_ideal']}")
    
    def _compute_cf_metrics(self):
        T = self.config['training_sessions']
        
        omega_base = np.sum(self.test_metrics['alpha_base_sessions'][1:])#skip the first session like in the paper
        omega_base /= ((T-1) * self.test_metrics['alpha_ideal'])
        
        omega_new = np.sum(self.test_metrics['alpha_new_sessions'])
        omega_new /= (T-1)
        
        omega_all = np.sum(self.test_metrics['alpha_all_sessions'])
        omega_all /= ((T-1)  * self.test_metrics['alpha_ideal'])
        
        self.cf_metrics['omega_base'] = omega_base
        self.cf_metrics['omega_new'] = omega_new
        self.cf_metrics['omega_all'] = omega_all
        
        logging.info("CF Metrics:")
        logging.info(f"Omega Base: {omega_base}")
        logging.info(f"Omega New: {omega_new}")
        logging.info(f"Omega All: {omega_all}")


    def _split(self, dataset, classes): 
        indices = []
        for i in range(len(dataset)):
            if dataset.targets[i] in classes:
                indices.append(i)
        return torch.utils.data.Subset(dataset, indices)

    def train(self):
        current_checkpoint = None
        incremental_classes_total = self.config['incremental_classes_total']
        incremental_classes_per_session = self.config['incremental_classes_per_session']
        base_classes = self.config['base_classes']
        T=self.config['training_sessions']

        base_train_dt = self._split(self.full_train_dt, base_classes)
        base_test_dt = self._split(self.full_test_dt, base_classes)
        
        incremental_classes_cumulative = []
        incremental_train_dt = None
        incremental_current_session_test_dt = None
        incremental_all_sessions_test_dt = None
        
        for session_number in range(T):
            logging.info(f"Training session {session_number} of {T} started.")
            
            new_checkpoint = os.path.join(self.checkpoints_path, f"{self.experiment_name}_session_{session_number}.ckpt")
            
            if session_number > 0:
                #draw incremental classes for this session only. these classes have not been used in previous sessions!
                #incremental classes may contain just 1 class (defined by incremental_classes_per_session)
                incremental_classes_total = list(set(incremental_classes_total) - set(incremental_classes_cumulative))
                #decrease the available incremental classes with each session
                incremental_classes = np.random.choice( 
                    incremental_classes_total, 
                    incremental_classes_per_session, 
                    replace=False
                ).tolist()
                
                #add the new incremental classes to the list of incremental classes used
                incremental_classes_cumulative.extend(incremental_classes)
                
                #train set for this session
                incremental_train_dt = self._split(self.full_train_dt, incremental_classes)
                
                #this test set is used for alpha_new(i), which is the test accuracy for session i immediately after it is learned
                incremental_current_session_test_dt = self._split(self.full_test_dt, incremental_classes)
                
                #this test set is used for alpha_all(i), which is the test accuracy of all of the test data for all classes seen to this point
                incremental_all_sessions_test_dt = self._split(self.full_test_dt, incremental_classes_cumulative)
                
                #decimate incremental train_dt to config['incremental_training_size']
                incremental_train_dt = torch.utils.data.Subset(incremental_train_dt, np.random.choice(len(incremental_train_dt), self.config['incremental_training_size'], replace=False))
                
                logging.info(f"current session classes: {incremental_classes} (num train samples: {len(incremental_train_dt)} / num test samples: {len(incremental_current_session_test_dt)})")
                logging.info(f"all session classes seen so far: {incremental_classes_cumulative} (no train samples for cumulative / num test samples: {len(incremental_all_sessions_test_dt)})")
                
            # fit model
            self.train_session(
                base_train_dt, incremental_train_dt, 
                current_checkpoint, new_checkpoint
            )
                        
            # test model and save metrics
            self.test_session(
                base_test_dt, incremental_current_session_test_dt, incremental_all_sessions_test_dt
            )
            
            if session_number == 0:
                self.test_metrics['session_classes'].append(base_classes)
            else:
                self.test_metrics['session_classes'].append(incremental_classes)
            
            current_checkpoint = new_checkpoint
        
        #compute cf metrics
        self._compute_cf_metrics()

    
    # train a single session
    def train_session(self, base_dt: Dataset, incremental_dt: Dataset, starting_checkpoint_path: str, new_checkpoint_path: str):
        if starting_checkpoint_path:
            logging.info(f"Finetuning model from checkpoint {starting_checkpoint_path}")
            self.session_trainer.init_model(starting_checkpoint_path)
            
            # draw replay data from base_dt
            replay_indices = np.random.choice(len(base_dt), self.config['replay_buffer_size'], replace=False)
            replay_dt = torch.utils.data.Subset(base_dt, replay_indices)
            
            self.session_trainer.fit(incremental_dt, replay_dt)
        
        else:
            logging.info(f"Training base model.")
            self.session_trainer.fit(base_dt)
        
        #save checkpoint
        self.session_trainer.save_model(new_checkpoint_path)
        
        
    def test_session(self, base_dt: Dataset, incremental_dt: Dataset, incremental_all_dt: Dataset):
        alpha_new = 0
        alpha_all = 0
        alpha_base = self.session_trainer.test(base_dt)
        
        if (incremental_dt is not None and incremental_all_dt is not None):
            logging.info(f"[base] Test: {alpha_base}")
            
            alpha_new = self.session_trainer.test(incremental_dt)
            logging.info(f"[incremental] Test: {alpha_new}")
            
            base_and_incremental_cumulative_dt = torch.utils.data.ConcatDataset([base_dt, incremental_all_dt])
            alpha_all = self.session_trainer.test(base_and_incremental_cumulative_dt)
            logging.info(f"[base+incremental] Test: {alpha_all}")
        
        self.test_metrics['alpha_new_sessions'].append(alpha_new)
        self.test_metrics['alpha_all_sessions'].append(alpha_all)
        self.test_metrics['alpha_base_sessions'].append(alpha_base)
        
    
    
    def save_metrics(self):
        #create folder for results
        exp_folder = f"incremental_trainer_experiments/{self.experiment_dir}/{self.experiment_name}"
        
        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)
            
        def json_serializer(obj):
            if isinstance(obj, type):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        # save session hyperparameters
        with open(f"{exp_folder}/session_hyperparams.json", 'w') as f:
            json.dump(self.session_trainer.hyperparams, f, default=json_serializer)
        
        # save trainer hyperparameters
        with open(f"{exp_folder}/config.json", 'w') as f:
            json.dump(self.config, f, default=json_serializer)
        
        # save metrics
        with open(f"{exp_folder}/metrics.json", 'w') as f:
            json.dump(self.test_metrics, f, default=json_serializer)
        
        # save catastrophic forgetting metrics
        with open(f"{exp_folder}/cf_metrics.json", 'w') as f:
            json.dump(self.cf_metrics, f, default=json_serializer)
        
        plt.figure()
        #plot alpha_base_sessions, alpha_new_sessions, alpha_all_sessions and save fig
        plt.plot(self.test_metrics['alpha_base_sessions'], label='Alpha Base Sessions')
        plt.plot(self.test_metrics['alpha_new_sessions'], label='Alpha New Sessions')
        plt.plot(self.test_metrics['alpha_all_sessions'], label='Alpha All Sessions')
        plt.legend()
        plt.xlabel('Training Sessions')
        
        plt.savefig(f"{exp_folder}/metrics.png")
        plt.close()
import torch
import copy
import random
import math
import torch.multiprocessing as mp
import torch.nn as nn
import time

cupy_imported = False
try :
    import cupy as cp
    cupy_imported = True
except ImportError:
    pass

class GeneticAlgorithmNN:
    def __init__(self, models=[], importances=[], mutation_scale=0.1, mutation_rate=0.1, crossover_rate=0.5, crossover_strategy = "random", model_args=[]):
        self.population = models
        self.importances = importances
        
        for model in self.population:
            model.origin = "initial"
        
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.crossover_strategy = crossover_strategy
        self.mutation_scale = mutation_scale
        self.model_class = models[0].__class__
        self.model_args = model_args
        
        assert len(self.population) == len(self.importances), "Models and importances must have the same length"
    
    def initialize_population(self, population_size):
        if len(self.model_args) == 0:
            return [self.model_class() for _ in range(population_size)]
        return [self.model_class(*self.model_args) for _ in range(population_size)]

    def mutate(self, model):
        mutated_model = copy.deepcopy(model)
        mutated_model.origin = "mutant"
        with torch.no_grad():
            for param in mutated_model.parameters():
                if cupy_imported:
                    param_data = param.data.cpu().numpy()
                    param_cp = cp.asarray(param_data)

                    mask = cp.random.rand(*param_cp.shape) < self.mutation_rate
                    mutation_values = cp.random.normal(loc=0.0, scale=self.mutation_scale, size=param_cp.shape)
                    param_cp[mask] += mutation_values[mask]

                    param.data = torch.tensor(cp.asnumpy(param_cp), device=param.device)
                else:
                    mask = torch.rand_like(param) < self.mutation_rate
                    mutation_values = torch.normal(mean=0.0, std=self.mutation_scale, size=param.shape).to(param.device)
                    param[mask] += mutation_values[mask]
        return mutated_model


    def crossover(self, parent1, parent2):
        parent1, parent2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
        parent1.origin = "child"
        parent2.origin = "child"
        
        if self.crossover_strategy == "importance":
            return self.crossover_importance(parent1, parent2)
        elif self.crossover_strategy == "random":
            return self.crossover_random(parent1, parent2)
        else:
            return parent1, parent2
    
    def crossover_importance(self, parent1, parent2, target_layers=["conv2.weight", "conv2.bias"]):
        base_importance = self.importances[0]  # Importance scores for the base dataset
        i = 0  # Counter for weights swapped
        total_weights = 0  # Counter for total weights in the target layer

        with torch.no_grad():
            for (name1, p1), (name2, p2) in zip(parent1.named_parameters(), parent2.named_parameters()):
                #ONLY target layer!!
                if name1 in target_layers and name1 in base_importance:
                    base_imp = base_importance[name1]
                    low_base_importance_mask = (base_imp < base_imp.mean()).cpu() #TODO

                    #changing ONLY non-critical weights
                    mask = torch.rand_like(p1) < self.crossover_rate
                    mask = mask & low_base_importance_mask

                    #i += torch.sum(mask).item()
                    #total_weights += mask.numel()

                    temp = p1.clone()
                    p1[mask] = p2[mask]
                    p2[mask] = temp[mask]

        #print(f"Layer: {target_layers}, Total weights: {total_weights}, Swapped weights: {i}")
        return parent1, parent2

    def crossover_random(self, parent1, parent2):
        with torch.no_grad():
            for p1, p2 in zip(parent1.parameters(), parent2.parameters()):
                if cupy_imported:
                    p1_data, p2_data = p1.data.cpu().numpy(), p2.data.cpu().numpy()
                    p1_cp, p2_cp = cp.asarray(p1_data), cp.asarray(p2_data)

                    mask = cp.random.rand(*p1_cp.shape) < self.crossover_rate
                    temp = cp.copy(p1_cp)
                    p1_cp[mask] = p2_cp[mask]
                    p2_cp[mask] = temp[mask]

                    p1.data = torch.tensor(cp.asnumpy(p1_cp), device=p1.device)
                    p2.data = torch.tensor(cp.asnumpy(p2_cp), device=p2.device)
                else:
                    mask = torch.rand_like(p1) < self.crossover_rate
                    temp = p1.clone()
                    p1[mask] = p2[mask]
                    p2[mask] = temp[mask]
        return parent1, parent2
    
    def add_mutants(self, model): 
        mutated_model = self.mutate(model)
        self.population.append(mutated_model)

    def add_children(self, parent1, parent2):
        child1, child2 = self.crossover(parent1, parent2)
        self.population.extend([child1, child2])

    
    def evaluate_population(self, test_loader):
        # parallel evaluation
        batched_population = nn.ModuleList(self.population).to("cuda")
        for model in batched_population:
            model.eval()

        fitness_values = [0] * len(self.population)
        sample_counts = [0] * len(self.population)

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to("cuda"), labels.to("cuda")

                outputs = [model(images) for model in batched_population]
                
                #accuracy calced for each model
                for i, output in enumerate(outputs):
                    _, preds = output.max(1)
                    correct = (preds == labels).sum().item()
                    fitness_values[i] += correct
                    sample_counts[i] += labels.size(0)

                #do this otherwise segfault
                del images, labels, outputs
                torch.cuda.empty_cache()

        fitness_values = [correct / total for correct, total in zip(fitness_values, sample_counts)]

        #risk of segfault if not done
        batched_population.to("cpu") # required otherwise segfault
        torch.cuda.empty_cache()  # Clear GPU memory

        return fitness_values

    def pareto_front_selection(self, population, incremental_fitness, replay_fitness):
        population_size = len(population)
        domination_counts = torch.zeros(population_size)
        dominated_sets = [[] for _ in range(population_size)]

        for i in range(population_size):
            for j in range(population_size):
                if i != j:
                    if (incremental_fitness[i] >= incremental_fitness[j] and
                        replay_fitness[i] >= replay_fitness[j]) and \
                        (incremental_fitness[i] > incremental_fitness[j] or
                         replay_fitness[i] > replay_fitness[j]):
                        # i dominates j
                        domination_counts[j] += 1
                        dominated_sets[i].append(j)

        # select individuals with no dominations, ie pareto front
        pareto_front = [idx for idx, count in enumerate(domination_counts) if count == 0]

        return pareto_front
    
    def select_parents(self, population, incremental_fitness_values, replay_fitness_values, combined_fitness_values, strategy="combined"):
        
        if strategy == "combined":
            total_fitness = sum(combined_fitness_values)
            probabilities = [f / total_fitness for f in combined_fitness_values]
            parents = random.choices(population, weights=probabilities, k=2)
        elif strategy == "pareto":
            pareto_front = self.pareto_front_selection(population, incremental_fitness_values, replay_fitness_values)
            parents = random.choices([population[idx] for idx in pareto_front], k=2)
        else:
            raise ValueError(f"Invalid selection strategy: {strategy}")
        
        return parents
        
    def evolve(self, incremental_train_loader, base_replay_train_loader, num_generations, selection_ratio=[0.5, 0.2, 0.2, 0.1],
               recall_importance=0.5, parent_selection_strategy="combined", initial_population_size=10):
        """
        recall_importance: [0,1] float that determines the importance of the ability to remember of base classes
        selection_ratio: list of 4 floats that sum to 1.0
        The first element is the ratio of the population that is the children of the previous generation
        The second element is the ratio of the population that is mutants of the previous generation
        The third element is the ratio of the population that is the top models of the previous generation
        The fourth element is the ratio of the population that is new models

        Notice that the number of generated models is rounded to the greatest smaller integer
        If the sum of the generated models is less than the population size, the remaining models are generated randomly
        """

        assert math.isclose(sum(selection_ratio), 1.0), f"The sum of the selection_ratio must be 1.0, got {sum(selection_ratio)}"
        
        #add initial mutants
        m=int((initial_population_size*0.8))
        for i in range(m//2):
            self.add_mutants(self.population[0])
            self.add_mutants(self.population[1])

        #fill the rest of the population with base models
        self.population.extend(self.initialize_population(initial_population_size - len(self.population)))
        print(f"Initial Population Size: {len(self.population)} with {m} mutants")
        print(f"Parent selection strategy: {parent_selection_strategy}")
        for i in range(num_generations):
            start_time = time.time()
            fitness_values_incremental = self.evaluate_population(incremental_train_loader)
            fitness_values_replay = self.evaluate_population(base_replay_train_loader)
            fitness_values_combined = [(a*(1-recall_importance) + b*recall_importance) for a, b in zip(fitness_values_incremental, fitness_values_replay)]
            #print(f"Generation {i}, Best Fitness Inc,Base,Combined:{max(fitness_values_incremental)},{max(fitness_values_replay)},{max(fitness_values_combined)}")
            
            #get indexes of best model in incremental, replay and combined fitness
            best_incremental_idx = fitness_values_incremental.index(max(fitness_values_incremental))
            best_replay_idx = fitness_values_replay.index(max(fitness_values_replay))
            best_combined_idx = fitness_values_combined.index(max(fitness_values_combined))
            
            #print origins of best models
            #print(f"Generation {i}, Best Models Origin Inc,Base,Combined: {self.population[best_incremental_idx].origin},{self.population[best_replay_idx].origin},{self.population[best_combined_idx].origin}")
            
            # old_population = self.population
            #old_population = [ model for _, model in sorted(zip(fitness_values_train, self.population), reverse=True) ]
            old_population = [model for _, model in sorted(zip(fitness_values_combined, self.population), key=lambda x: x[0], reverse=True)]
            old_population_size = len(old_population)
            self.population = []

            # Children
            num_children = int(selection_ratio[0] * old_population_size)
            if num_children % 2 == 1: # if odd, decrement to make it even
                num_children -= 1
            for _ in range(num_children//2):
                parent1, parent2 = self.select_parents(old_population, 
                                                       fitness_values_incremental, 
                                                       fitness_values_replay,
                                                       fitness_values_combined,
                                                       strategy=parent_selection_strategy)
                self.add_children(parent1, parent2)
            
            # Mutants
            num_mutants = int(selection_ratio[1] * old_population_size)
            for _ in range(num_mutants):
                model = random.choice(old_population)
                self.add_mutants(model)

            # Top models
            num_top_models = int(selection_ratio[2] * old_population_size)
            for model in old_population[:num_top_models]:
                model.origin = "top"  # Tag as top model
                self.population.append(model)

            # New models
            num_new_models = old_population_size - len(self.population)
            for _ in range(num_new_models):
                if len(self.model_args) == 0:
                    new_model = self.model_class()
                else:
                    new_model = self.model_class(*self.model_args)
                
                new_model.origin = "new"
                self.population.append(new_model)
                
            #print(f"Generation {i}, Mutants,Children,Top,New: {num_mutants},{num_children},{num_top_models},{num_new_models}")
            print(f"Generation {i} time", round(time.time() - start_time,2))

        best_model = self.population[fitness_values_combined.index(max(fitness_values_combined))]

        return best_model

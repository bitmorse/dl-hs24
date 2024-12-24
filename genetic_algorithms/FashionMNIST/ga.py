import torch
import copy
import random
import math

cupy_imported = False
try :
    import cupy as cp
    cupy_imported = True
except ImportError:
    pass

class GeneticAlgorithmNN:
    def __init__(self, models, mutation_rate=0.1, crossover_rate=0.5, model_args=[]):
        self.population = models
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.model_class = models[0].__class__
        self.model_args = model_args

    def initialize_population(self, population_size):
        if len(self.model_args) == 0:
            return [self.model_class() for _ in range(population_size)]
        return [self.model_class(*self.model_args) for _ in range(population_size)]

    def mutate(self, model):
        mutated_model = copy.deepcopy(model)
        with torch.no_grad():
            for param in mutated_model.parameters():
                if cupy_imported:
                    param_data = param.data.cpu().numpy()
                    param_cp = cp.asarray(param_data)

                    mask = cp.random.rand(*param_cp.shape) < self.mutation_rate
                    mutation_values = cp.random.normal(loc=0.0, scale=0.1, size=param_cp.shape)
                    param_cp[mask] += mutation_values[mask]

                    param.data = torch.tensor(cp.asnumpy(param_cp), device=param.device)
                else:
                    mask = torch.rand_like(param) < self.mutation_rate
                    mutation_values = torch.normal(mean=0.0, std=0.1, size=param.shape).to(param.device)
                    param[mask] += mutation_values[mask]
        return mutated_model

    def crossover(self, parent1, parent2):
        child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
        with torch.no_grad():
            for p1, p2 in zip(child1.parameters(), child2.parameters()):
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
        return child1, child2
    
    def add_mutants(self, model): 
        mutated_model = self.mutate(model)
        self.population.append(mutated_model)

    def add_children(self, parent1, parent2):
        child1, child2 = self.crossover(parent1, parent2)
        self.population.extend([child1, child2])

    def evaluate_population(self, test_loader):
        fitness_values = []
        for model in self.population:
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
            fitness_values.append(acc)

            model.to("cpu")

        return fitness_values

    def select_parents(self, population, fitness_values):
        total_fitness = sum(fitness_values)
        probabilities = [f / total_fitness for f in fitness_values]
        parents = random.choices(population, weights=probabilities, k=2)
        return parents
    
    def evolve(self, train_loader, test_loader, num_generations, selection_ratio=[0.5, 0.2, 0.2, 0.1]):
        """
        selection_ratio: list of 4 floats that sum to 1.0
        The first element is the ratio of the population that is the children of the previous generation
        The second element is the ratio of the population that is mutants of the previous generation
        The third element is the ratio of the population that is the top models of the previous generation
        The fourth element is the ratio of the population that is new models

        Notice that the number of generated models is rounded to the greatest smaller integer
        If the sum of the generated models is less than the population size, the remaining models are generated randomly
        """

        assert math.isclose(sum(selection_ratio), 1.0), f"The sum of the selection_ratio must be 1.0, got {sum(selection_ratio)}"

        for i in range(num_generations):
            fitness_values_train = self.evaluate_population(train_loader)
            print(f"Generation {i}, Best Train Fitness: {max(fitness_values_train)}")

            fitness_values_test = self.evaluate_population(test_loader)
            print(f"Generation {i}, Best Test Fitness: {max(fitness_values_test)}")

            # old_population = self.population
            #old_population = [ model for _, model in sorted(zip(fitness_values_train, self.population), reverse=True) ]
            old_population = [model for _, model in sorted(zip(fitness_values_train, self.population), key=lambda x: x[0], reverse=True)]

            self.population = []

            # Children
            num_children = int(selection_ratio[0] * len(old_population)) % 2 == 0
            if num_children % 2 == 1:
                num_children -= 1
            for _ in range(num_children//2):
                parent1, parent2 = self.select_parents(old_population, fitness_values_train)
                self.add_children(parent1, parent2)

            # Mutants
            num_mutants = int(selection_ratio[1] * len(old_population))
            for _ in range(num_mutants):
                model = random.choice(old_population)
                self.add_mutants(model)

            # Top models
            num_top_models = int(selection_ratio[2] * len(old_population))
            self.population.extend(old_population[:num_top_models])

            # New models
            for _ in range(len(old_population) - len(self.population)):
                if len(self.model_args) == 0:
                    self.population.append(self.model_class())
                else:
                    self.population.append(self.model_class(*self.model_args))

        best_model = self.population[fitness_values_train.index(max(fitness_values_train))]
        return best_model

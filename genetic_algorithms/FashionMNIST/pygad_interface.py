from pygad import GA
import torch
import numpy as np
from genetic_algorithms.FashionMNIST.model import test_ann

# Convert weights and biases to a 1D array.
def model_weights_as_vector(model):
    weights_vector = []

    for curr_weights in model.state_dict().values():
        curr_weights = curr_weights.cpu().detach().numpy()
        vector = np.reshape(curr_weights, newshape=(curr_weights.size))
        weights_vector.extend(vector)

    return np.array(weights_vector)

# Revert the weights and biases to the original model architecture.
def model_weights_as_dict(model, weights_vector):
    weights_dict = model.state_dict()

    start = 0
    for key in weights_dict:
        w_matrix = weights_dict[key].cpu().detach().numpy()
        layer_weights_shape = w_matrix.shape
        layer_weights_size = w_matrix.size

        layer_weights_vector = weights_vector[start:start + layer_weights_size]
        layer_weights_matrix = np.reshape(layer_weights_vector, newshape=(layer_weights_shape))
        weights_dict[key] = torch.from_numpy(layer_weights_matrix).to("cuda")

        start = start + layer_weights_size

    return weights_dict

def fitness_func(ga_instanse, solution, sol_idx):
    pass
    # model = ga_instanse.model().to("cuda")
    # model_weights = model_weights_as_dict(model, solution)
    # model.load_state_dict(model_weights)

    # loss = torch.nn.CrossEntropyLoss()

    # model.eval()
    # with torch.no_grad():
    #     loss_val = 0.0
    #     for i, (images, labels) in enumerate(iter(ga_instanse.trainloader)):
    #         images, labels = images.to("cuda"), labels.to("cuda")
    #         out = model(images)
    #         loss_val += loss(out, labels).item()

    # solution_fitness = 1.0 / (loss_val + 1e-8)

    # del model

    # return solution_fitness

def on_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")

    model = ga_instance.best_solution_NN()
    print(f"Test with the training set: {test_ann(model, ga_instance.train_loader, gpu2cpu=False):.4f}")
    print()

    del model

def init_population(population_size, models: list):
    initial_population = []
    for i in range(population_size):
        model_idx = i % len(models)
        model = models[model_idx]
        model_weights = model_weights_as_vector(model)
        initial_population.append(model_weights)
                                  
    return initial_population

class PyGADNN(GA):
    def __init__(self, model:callable=None, train_loader=None, **kwargs):
        self.model = model
        self.train_loader = train_loader
        super().__init__(**kwargs)

        assert self.sol_per_pop % self.fitness_batch_size == 0, "sol_per_pop should be divisible by fitness_batch_size"
        self.nn_population = torch.nn.ModuleList([ model() for _ in range(self.fitness_batch_size) ])
        if self.sol_per_pop == self.fitness_batch_size:
            self.nn_population.to("cuda")

    def cal_pop_fitness(self):
        # 'last_generation_parents_as_list' is the list version of 'self.last_generation_parents'
        # It is used to return the parent index using the 'in' membership operator of Python lists. This is much faster than using 'numpy.where()'.
        if self.last_generation_parents is not None:
            last_generation_parents_as_list = [
                list(gen_parent) for gen_parent in self.last_generation_parents]

        # 'last_generation_elitism_as_list' is the list version of 'self.last_generation_elitism'
        # It is used to return the elitism index using the 'in' membership operator of Python lists. This is much faster than using 'numpy.where()'.
        if self.last_generation_elitism is not None:
            last_generation_elitism_as_list = [
                list(gen_elitism) for gen_elitism in self.last_generation_elitism]

        pop_fitness = []

        # Calculating the fitness value of each solution in the current population.
        for batch in range(0, len(self.population), self.fitness_batch_size):
            # Get the solutions in the current batch.
            solutions = self.population[batch:batch + self.fitness_batch_size]

            if self.sol_per_pop != self.fitness_batch_size:
                self.nn_population.to("cuda")
            for model, solution in zip(self.nn_population, solutions):
                model_weights = model_weights_as_dict(model, solution)
                model.load_state_dict(model_weights)
                model.eval()
            
            loss = torch.nn.CrossEntropyLoss()
            batch_fitness = [ 0.0 for _ in range(self.fitness_batch_size) ]

            with torch.no_grad():
                for images, labels in self.train_loader:
                    images, labels = images.to("cuda"), labels.to("cuda")

                    outputs = [model(images) for model in self.nn_population]

                    for i, output in enumerate(outputs):
                        batch_fitness[i] += loss(output, labels).item()
            
            pop_fitness.extend(batch_fitness)
            if self.sol_per_pop != self.fitness_batch_size:
                self.nn_population.to("cpu")

        pop_fitness = np.array(pop_fitness)
        pop_fitness = 1.0 / (pop_fitness + 1e-8)

        return pop_fitness
    
    def best_solution_NN(self):
        solution, solution_fitness, solution_idx = self.best_solution()
        model = self.model().to("cuda")
        model_weights = model_weights_as_dict(model, solution)
        model.load_state_dict(model_weights)
        return model
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
    model_weights = model_weights_as_dict(ga_instanse.model, solution)
    ga_instanse.model.load_state_dict(model_weights)

    loss = torch.nn.CrossEntropyLoss()

    ga_instanse.model.eval()
    with torch.no_grad():
        loss_val = 0.0
        for i, (images, labels) in enumerate(iter(ga_instanse.trainloader)):
            images, labels = images.to("cuda"), labels.to("cuda")
            out = ga_instanse.model(images)
            loss_val += loss(out, labels).item()

    solution_fitness = 1.0 / (loss_val+ 1e-8)

    return solution_fitness

def on_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    model_weights = model_weights_as_dict(ga_instance.model, solution)
    ga_instance.model.load_state_dict(model_weights)
    print(f"Test with the training set: {test_ann(ga_instance.model, ga_instance.trainloader, gpu2cpu=False):.4f}")
    print()

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
        self.model = model().to("cuda")
        self.trainloader = train_loader
        super().__init__(**kwargs)
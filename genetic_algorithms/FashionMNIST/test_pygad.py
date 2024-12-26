from data import split, select
from model import ANN, train_ann, test_ann

import torch
from torchvision import datasets, transforms

import numpy as np

batch_size = 64
dataset_name = 'FashionMNIST'
data_path=f'/scratch/zyi/codeSpace/data/{dataset_name}'
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])
train_dt = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
test_dt = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)

train_p1, train_p2 = split(train_dt, [0, 1, 2, 3, 4]), split(train_dt, [5, 6, 7, 8, 9])
mini_train = select(train_dt, 0.01)

train_loader = torch.utils.data.DataLoader(train_dt, batch_size=batch_size, shuffle=True)
train_loader_p1 = torch.utils.data.DataLoader(train_p1, batch_size=batch_size, shuffle=True)
train_loader_p2 = torch.utils.data.DataLoader(train_p2, batch_size=batch_size, shuffle=True)
mini_train_loader = torch.utils.data.DataLoader(mini_train, batch_size=len(mini_train), shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dt, batch_size=batch_size, shuffle=False)

net1 = ANN().to("cuda")
net2 = ANN().to("cuda")

optimizer1 = torch.optim.Adam(net1.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(net2.parameters(), lr=0.001)

criterion = torch.nn.CrossEntropyLoss()
train_ann(net1, train_loader_p1, criterion, optimizer1, 1, gpu2cpu=False)
train_ann(net2, train_loader_p2, criterion, optimizer2, 1, gpu2cpu=False)

print(f"Testing model 1: {test_ann(net1, test_loader)}")
print(f"Testing model 2: {test_ann(net2, test_loader)}")

import pygad

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

model = ANN().to("cuda")

# Fitness function
def fitness_func(ga_instanse, solution, sol_idx):
    model_weights = model_weights_as_dict(model, solution)
    model.load_state_dict(model_weights)
    # model.to("cuda")

    loss = torch.nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        loss_val = 0.0
        for i, (images, labels) in enumerate(iter(mini_train_loader)):
            images, labels = images.to("cuda"), labels.to("cuda")
            out = model(images)
            loss_val += loss(out, labels).item()

    solution_fitness = 1.0 / (loss_val+ 1e-8)

    return solution_fitness

# Output per generation
def on_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    model_weights = model_weights_as_dict(model, solution)
    model.load_state_dict(model_weights)
    # model.to("cuda")
    print(f"Test with the training set: {test_ann(model, mini_train_loader, gpu2cpu=False):.4f}")
    print(f"Test with the test set: {test_ann(model, test_loader, gpu2cpu=False):.4f}")
    print()

# Create initial population
def init_population(population_size, models: list):
    initial_population = []
    for i in range(population_size):
        model_idx = i % len(models)
        model = models[model_idx]
        model_weights = model_weights_as_vector(model)
        initial_population.append(model_weights)
                                  
    return initial_population

# Create the PyGAD instance

models = [net1, net2]
population_size = 100
initial_population = init_population(population_size, models)

ga_instance = pygad.GA(num_generations=1000,
                       num_parents_mating=5,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=on_generation,
                       sol_per_pop=population_size,
                       parent_selection_type="sss",
                       keep_parents=-1,
                       K_tournament=3,
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_percent_genes=10.0,
                       mutation_by_replacement=False,
                       random_mutation_min_val=-0.1,
                       random_mutation_max_val=0.1)

# Start the evolution
ga_instance.run()

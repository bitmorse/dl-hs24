from torchvision import datasets, transforms

INCREMENTAL_TRAINER_CONFIG = {
    'replay_buffer_size': 1000, #TODO: show how more data makes GA perform worse
    'incremental_training_size': 1000, #TODO: show how more data makes GA perform worse
    'training_sessions': 6,
    'base_classes': [0,1,2,3,4],
    'incremental_classes_total': [5,6,7,8,9],
    'incremental_classes_per_session': 1,
    'dataset_name': 'FashionMNIST'
}

def get_datasets(data_path='/tmp'):
    data_path+=f"/{INCREMENTAL_TRAINER_CONFIG['dataset_name']}"
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    # select dataset, note that the model is the same for all datasets currently. CIFAR10 is tranformed to grayscale!
    if INCREMENTAL_TRAINER_CONFIG['dataset_name'] == 'FashionMNIST':
        train_dt = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
        test_dt = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
    elif INCREMENTAL_TRAINER_CONFIG['dataset_name'] == 'CIFAR10':
        train_dt = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        test_dt = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
    
    return train_dt, test_dt

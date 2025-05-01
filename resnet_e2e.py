import numpy as np
import pandas as pd #type: ignore
import matplotlib.pyplot as plt
from tqdm import tqdm  #type: ignore
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

import resnet_models
import trainer


def get_device(index):
    device = torch.device(f'cuda:{index}') if torch.cuda.is_available() else torch.device('cpu')
    return device

def set_seed(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p = 0.5),
        #transforms.Pad(padding = 4),
        transforms.RandomCrop(size = (32,32), padding = 4),
        transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.247, 0.243, 0.261])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.247, 0.243, 0.261])
    ])

    return train_transform, test_transform


def download_datasets(dwd_dir, train_transform, test_transform):
    # Get training and testing data
    train_data = datasets.CIFAR10(
        root= dwd_dir, # where to download data to?
        train=True, # get training data
        transform = train_transform,
        download=True, # download data if it doesn't exist on disk
    )

    test_data = datasets.CIFAR10(
        root= dwd_dir,
        train=False, # get test data
        transform = test_transform,
        download=True,
    )

    return train_data, test_data

def get_config():
    config = dict(

        BATCH_SIZE = 64,
        LR = 0.001,
        MOMENTUM = 0.9,
        INPUT_CHANNEL = 3,
        INITIAL_FEATURE_DEPTH = 32,
        FINAL_FEATURE_DEPTH = 128
    )

    return config


def get_dataloader(dataset, batchsize = 64, shuffle = True):
    dataloader = DataLoader(dataset = dataset, batch_size= batchsize, shuffle= shuffle)
    return dataloader


def create_e2e_model(model_config,
                     layer_ids_to_be_added,
                     device):

    INPUT_CHANNEL = model_config['INPUT_CHANNEL']
    INITIAL_FEATURE_DEPTH = model_config['INITIAL_FEATURE_DEPTH']

    resnet = resnet_models.greedy_resnet(INPUT_CHANNEL, 
                                         INITIAL_FEATURE_DEPTH, 
                                         INITIAL_FEATURE_DEPTH, 
                                         INITIAL_FEATURE_DEPTH, 
                                         10).to(device)
    
    # Add all the layers at once
    in_features = INITIAL_FEATURE_DEPTH

    for layer_id in layer_ids_to_be_added:
        if layer_id == 0:
            resnet_models.add_layers(resnet, in_features, in_features, layer_id, in_features, device)
        else:
            out_features = 2*in_features
            resnet_models.add_layers(resnet, in_features, out_features, layer_id, in_features, device)
            in_features = out_features
    
    return resnet
    


if __name__ == "__main__":

    # Get current date and time
    start = datetime.now()

    # Format with date and time
    print("Started the Model Training at:", start.strftime("%Y-%m-%d %H:%M:%S"))
    
    device = get_device(6)
    set_seed(42)
    print('Running on device: ', device)

    # Get the transforms
    train_transform, test_transform = get_transforms()

    # Get the dataset
    print('Dataset download in progress...')
    train_data, test_data = download_datasets(dwd_dir= '/data/home/samsadalam/MLSP_Project/torch_data_CIFAR',
                      train_transform= train_transform,
                      test_transform= test_transform)
    print('\nDataset downloaded successfully.')
    
    # Get the config
    config = get_config()

    BATCH_SIZE = config['BATCH_SIZE']
    LR = config['LR']
    MOMENTUM = config['MOMENTUM']
    INPUT_CHANNEL = config['INPUT_CHANNEL']
    INITIAL_FEATURE_DEPTH = config['INITIAL_FEATURE_DEPTH']
    FINAL_FEATURE_DEPTH = config['FINAL_FEATURE_DEPTH']

    # Get the dataloader object
    print('\nCreating dataloader objects')
    train_dataloader = get_dataloader(train_data, batchsize = BATCH_SIZE, shuffle= True)
    test_dataloader = get_dataloader(test_data, batchsize= BATCH_SIZE, shuffle = False)

    e2e_training_params = dict(
        MAX_EPOCHS = 60,
        EPOCHS_TO_ADD_LAYER = 5,
        MAX_EPOCHS_TO_ADD_LAYER = 40,
        LR = 0.001,
        MOMENTUM = 0.9
    )
    
    e2e_metric = {
        'loss_history': {'train':[], 'val': []},
        'acc_history': {'train':[], 'val': []},
        'flops_history': [],
        'layer_ids_to_be_added': [0, 0, 0, 0, 1, 0, 0, 0, 0,  1, 0, 0, 0, 0, 1]
    }

    resnet = create_e2e_model(config, e2e_metric['layer_ids_to_be_added'], device)
    
    # Now define optimizer and criterion for the resnet model
    optimizer = torch.optim.SGD(resnet.parameters(), lr = LR, momentum = MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    print('\nSuccessfully created the greedy resnet model')
    print('\nStarting the e2e training...')

    e2e_training_metric = trainer.e2e_training(
                            model = resnet,
                            train_dataloader = train_dataloader,
                            test_dataloader = test_dataloader, 
                            optimizer = optimizer, 
                            criterion = criterion, 
                            initial_in_features = INITIAL_FEATURE_DEPTH, 
                            save_steps = 100,
                            device = device,
                            training_params = e2e_training_params,
                            **e2e_metric)

    test_acc = trainer.test_step(resnet, test_dataloader, trainer.Accuracy, device)
    print('Test Accuracy: ', test_acc)

    # Save the trained e2e model
    e2e_resnet_path = '/data/home/samsadalam/MLSP_Project/e2e_resnet_model.pth'
    torch.save(resnet.state_dict(), e2e_resnet_path)

    # Save the latest trained metric
    e2e_train_metric_save_dir = '/data/home/samsadalam/MLSP_Project/resnet_e2e_metric.pth'
    torch.save(e2e_training_metric, e2e_train_metric_save_dir)

    # Get current date and time
    end = datetime.now()

    # Format with date and time
    print("Finished the Model Training at:", end.strftime("%Y-%m-%d %H:%M:%S"))










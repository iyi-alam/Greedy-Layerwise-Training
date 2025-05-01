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
                     device):

    INPUT_CHANNEL = model_config['INPUT_CHANNEL']
    INITIAL_FEATURE_DEPTH = model_config['INITIAL_FEATURE_DEPTH']

    resnet = resnet_models.greedy_resnet(INPUT_CHANNEL, 
                                         INITIAL_FEATURE_DEPTH, 
                                         INITIAL_FEATURE_DEPTH, 
                                         INITIAL_FEATURE_DEPTH, 
                                         10).to(device)
    
    return resnet


if __name__ == "__main__":

    # Get current date and time
    start = datetime.now()

    # Format with date and time
    print("Started the Model Training at:", start.strftime("%Y-%m-%d %H:%M:%S"))
    
    device = get_device(1)
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

    # Create the greedy resnet model
    resnet = create_e2e_model(config, device)
    optimizer = torch.optim.SGD(resnet.parameters(), lr = LR, momentum = MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    print('\nSuccessfully created the greedy resnet model')

    greedy_training_params = dict(
        MAX_EPOCHS = 66,
        EPOCHS_TO_ADD_LAYER = 5,
        MAX_EPOCHS_TO_ADD_LAYER = 65,
        LR = 0.001,
        MOMENTUM = 0.9,
        FREEZE = False
    )
    
    greedy_metric = {
        'loss_history': {'train':[], 'val': []},
        'acc_history': {'train':[], 'val': []},
        'flops_history': [],
        'layer_ids_to_be_added': [0, 0, 0, 0, 1, 0, 0, 0, 0,  1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0]
    }
    
    print('\nInitiating the greedy training')
    FREEZE = greedy_training_params['FREEZE']

    greedy_train_matric = trainer.greed_layerwise_training(
        model = resnet,
        train_dataloader = train_dataloader,
        test_dataloader = test_dataloader, 
        optimizer = optimizer, 
        criterion = criterion, 
        initial_in_features = INITIAL_FEATURE_DEPTH, 
        save_steps = 100,
        add_layers = resnet_models.add_layers,
        update_optimizer = resnet_models.update_optimizer,
        device = device,
        training_params = greedy_training_params,
        **greedy_metric
    )

    # Test the model that has been trained layerwise
    test_acc = trainer.test_step(resnet, test_dataloader, trainer.Accuracy, device)
    print('Test Accuracy: ', test_acc)

    print('\nMoving for further static training of freezed greedy model...')

    #E2E training of the greedy model
    updated_optmizer = torch.optim.SGD(resnet.parameters(), lr = LR*0.1, momentum = MOMENTUM)
    if FREEZE:
        model_save_path = '/data/home/samsadalam/MLSP_Project/greedy_resnet_model_freeze.pth'
    else:
        model_save_path = '/data/home/samsadalam/MLSP_Project/greedy_resnet_model.pth'

    greedy_final_metric = trainer.static_training(
                model = resnet,
                train_dataloader = train_dataloader,
                test_dataloader = test_dataloader, 
                optimizer = updated_optmizer, 
                criterion = criterion, 
                initial_in_features = INITIAL_FEATURE_DEPTH, 
                save_steps = 100,
                device = device,
                max_epochs = 10,
                model_path = model_save_path,
                best_acc= test_acc,
                **greedy_train_matric)

    #print(greedy_train_matric['loss_history'])

    # Save the latest trained metric
    if FREEZE:
        greedy_train_metric_save_dir = '/data/home/samsadalam/MLSP_Project/resnet_greedy_metric_freeze.pth'
    else:
        greedy_train_metric_save_dir = '/data/home/samsadalam/MLSP_Project/resnet_greedy_metric.pth'
    torch.save(greedy_final_metric, greedy_train_metric_save_dir)

    # Save the best performing model
    print('Training Finished!')

    # Get current date and time
    end = datetime.now()

    # Format with date and time
    print("Finished the Model Training at:", end.strftime("%Y-%m-%d %H:%M:%S"))










import os
import shutil
from PIL import Image
import pandas as pd #type:ignore
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm  #type: ignore
from torch import nn
from datetime import datetime

import resnet_models
import trainer

def organize_val_folder(val_dir, create_again = True):
    """
    Organize TinyImageNet validation images into class-specific subfolders based on val_annotations.txt.
    
    Args:
        val_dir (str): Path to the validation folder containing images and val_annotations.txt
    """
    annotation_file = os.path.join(val_dir, 'val_annotations.txt')
    val_img_dir = os.path.join(val_dir, 'images')
    
    # Read the annotation file
    annotations = pd.read_csv(annotation_file, sep='\t', header=None,
                              names=['image', 'class', 'x', 'y', 'w', 'h'])
    
    # Create a folder for validation data organized by class
    val_class_dir = os.path.join(val_dir, 'val_organized')
    if not os.path.exists(val_class_dir):
        os.makedirs(val_class_dir)
    elif not create_again:
        return
    else:
        pass
    
    # Move images to class-specific subfolders
    for _, row in annotations.iterrows():
        img_name = row['image']
        class_id = row['class']
        class_dir = os.path.join(val_class_dir, class_id)
        
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        src_path = os.path.join(val_img_dir, img_name)
        dst_path = os.path.join(class_dir, img_name)
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
    
    print(f"Validation images organized into {val_class_dir}")


class TinyImageNetDataset(Dataset):
    """
    Custom PyTorch Dataset for TinyImageNet.
    
    Args:
        data_dir (str): Path to train or val_organized folder
        transform (callable, optional): Optional transform to be applied to images
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(data_dir) 
                            if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        
        print(f"Found {len(self.classes)} classes in {data_dir}")
        
        
        for cls in self.classes:
            class_dir = os.path.join(data_dir, cls)
            images_subdir = os.path.join(class_dir, 'images')
            
            if os.path.exists(images_subdir):
                img_files = [f for f in os.listdir(images_subdir) 
                            if f.lower().endswith(('.jpeg', '.jpg'))]
                #print(f"Class {cls} has {len(img_files)} images (from 'images' subfolder)")
                for img_name in img_files:
                    img_path = os.path.join(images_subdir, img_name)
                    self.images.append((img_path, self.class_to_idx[cls]))
            else:
                img_files = [f for f in os.listdir(class_dir) 
                            if f.lower().endswith(('.jpeg', '.jpg'))]
                #print(f"Class {cls} has {len(img_files)} images (directly in class folder)")
                for img_name in img_files:
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append((img_path, self.class_to_idx[cls]))
        
        if len(self.images) == 0:
            raise ValueError(f"No valid images found in {data_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms():
    """
    Define transformations for train and validation datasets.
    
    Returns:
        dict: Dictionary containing train and val transforms
    """
    return {
        'train': transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    }

# Step 4: Create Datasets and DataLoaders
def create_dataloaders(data_root, batch_size=64, organize_val = True):
    """
    Create PyTorch DataLoaders for TinyImageNet train and validation sets.
    
    Args:
        data_root (str): Path to the root directory containing train and val folders
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for DataLoader
    
    Returns:
        tuple: (train_loader, val_loader, class_to_idx)
    """
    # Organize validation folder
    val_dir = os.path.join(data_root, 'val')
    organize_val_folder(val_dir, create_again= organize_val)
    
    # Define paths
    train_dir = os.path.join(data_root, 'train')
    val_organized_dir = os.path.join(val_dir, 'val_organized')
    
    # Get transformations
    transforms_dict = get_transforms()
    
    # Create datasets
    train_dataset = TinyImageNetDataset(train_dir, transform=transforms_dict['train'])
    val_dataset = TinyImageNetDataset(val_organized_dir, transform=transforms_dict['val'])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, train_dataset.class_to_idx


def get_device(index):
    device = torch.device(f'cuda:{index}') if torch.cuda.is_available() else torch.device('cpu')
    return device

def set_seed(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def get_config(num_classes=10):
    config = dict(

        BATCH_SIZE = 64,
        LR = 0.1,
        MOMENTUM = 0.9,
        INPUT_CHANNEL = 3,
        INITIAL_FEATURE_DEPTH = 32,
        FINAL_FEATURE_DEPTH = 128,
        NUM_CLASSES = num_classes,
    )

    return config

def create_e2e_model(model_config,
                     device):

    INPUT_CHANNEL = model_config['INPUT_CHANNEL']
    INITIAL_FEATURE_DEPTH = model_config['INITIAL_FEATURE_DEPTH']
    NUM_CLASSES = model_config['NUM_CLASSES']

    resnet = resnet_models.greedy_resnet(INPUT_CHANNEL, 
                                         INITIAL_FEATURE_DEPTH, 
                                         INITIAL_FEATURE_DEPTH, 
                                         INITIAL_FEATURE_DEPTH, 
                                         NUM_CLASSES).to(device)
    
    return resnet

if __name__ == "__main__":
    # Replace with your TinyImageNet dataset path
    data_root = './tiny-imagenet-200'

    # Get the config
    config = get_config(num_classes= 10)

    BATCH_SIZE = config['BATCH_SIZE']
    LR = config['LR']
    MOMENTUM = config['MOMENTUM']
    INPUT_CHANNEL = config['INPUT_CHANNEL']
    INITIAL_FEATURE_DEPTH = config['INITIAL_FEATURE_DEPTH']
    FINAL_FEATURE_DEPTH = config['FINAL_FEATURE_DEPTH']
    RESUME = False
    CHECKPOINT = -1
    
    # Create dataloaders
    train_dataloader, test_dataloader, class_to_idx = create_dataloaders(data_root, batch_size=BATCH_SIZE, organize_val= False)
    
    # Print dataset sizes
    print(f"Number of training samples: {len(train_dataloader.dataset)}")
    print(f"Number of validation samples: {len(test_dataloader.dataset)}")
    print(f"Number of classes: {len(class_to_idx)}")
    
    # Example: Iterate through one batch
    for images, labels in train_dataloader:
        print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
        break

    # Get current date and time
    start = datetime.now()

    # Format with date and time
    print("Started the Model Training at:", start.strftime("%Y-%m-%d %H:%M:%S"))
    
    device = get_device(7)
    set_seed(42)
    print('Running on device: ', device)

    # Create the greedy resnet model
    config['NUM_CLASSES'] = len(class_to_idx)
    greedy_training_params = dict(
        MAX_EPOCHS = 100,
        EPOCHS_TO_ADD_LAYER = 10,
        MAX_EPOCHS_TO_ADD_LAYER = 210,
        LR = 0.05,
        MOMENTUM = 0.9,
        FREEZE = False,
        WARMUP_EPOCHS = 10,
        STEP_MILESTONES = [60, 90]
    )

    LR = greedy_training_params['LR']

    resnet = create_e2e_model(config, device)
    optimizer = torch.optim.SGD(resnet.parameters(), lr = LR, momentum = MOMENTUM)
    criterion = nn.CrossEntropyLoss()

    print('\nSuccessfully created the greedy resnet model')
    
    # # Add warmup and scheduler parameters to training_params
    # WARMUP_EPOCHS = training_params.get('WARMUP_EPOCHS', 5)  # Default to 5 epochs
    # STEP_MILESTONES = training_params.get('STEP_MILESTONES', [30, 60])  # Default milestones


    greedy_metric = {
        'loss_history': {'train':[], 'val': []},
        'acc_history': {'train':[], 'val': []},
        'flops_history': [],
        'last_epoch': -1,
        'best_acc': 0.0,
        'layer_ids_to_be_added': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 , 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    }

    print('\nInitiating the greedy training')
    FREEZE = greedy_training_params['FREEZE']

    # Load the checkpoint
    if FREEZE:
        checkpoint_dir = '/data/home/samsadalam/MLSP_Project/pth_files/imagenet_e2e_metric_freeze.pth'
        model_dir = '/data/home/samsadalam/MLSP_Project/pth_files/imagenet_e2e_model_freeze.pth'
    else:
        checkpoint_dir = '/data/home/samsadalam/MLSP_Project/pth_files/imagenet_e2e_metric.pth'
        model_dir = '/data/home/samsadalam/MLSP_Project/pth_files/imagenet_e2e_model.pth'

    # Resume Training
    if RESUME:
        saved_metric = torch.load(checkpoint_dir, weights_only= False)
        greedy_metric['loss_history'] = saved_metric['loss_history']
        greedy_metric['acc_history'] = saved_metric['acc_history']
        greedy_metric['flops_history'] = saved_metric['flops_history']
        greedy_metric['last_epoch'] = saved_metric['last_epoch']
        greedy_metric['best_acc'] = saved_metric['best_acc']


    greedy_train_matric = trainer.imagenet_e2e_training(
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
        num_classes= config['NUM_CLASSES'],
        use_tqdm= False,
        model_path = model_dir,
        metric_path = checkpoint_dir,
        use_scheduler= False,
        **greedy_metric
    )

    # Test the model that has been trained layerwise
    resnet.load_state_dict(torch.load(model_dir))
    test_acc = trainer.test_step(resnet, test_dataloader, trainer.Accuracy, device)
    print('Test Accuracy: ', test_acc)

    # Get current date and time
    end = datetime.now()

    # Format with date and time
    print("Finished the Model Training at:", end.strftime("%Y-%m-%d %H:%M:%S"))

    # Save the checkpoint
    # Save the latest trained metric
    # Load the checkpoint
    if FREEZE:
        checkpoint_dir = '/data/home/samsadalam/MLSP_Project/pth_files/imagenet_e2e_metric_freeze.pth'
        model_dir = '/data/home/samsadalam/MLSP_Project/pth_files/imagenet_e2e_model_freeze.pth'
    else:
        checkpoint_dir = '/data/home/samsadalam/MLSP_Project/pth_files/imagenet_e2e_metric.pth'
        model_dir = '/data/home/samsadalam/MLSP_Project/pth_files/imagenet_e2e_model.pth'
    torch.save(greedy_train_matric, checkpoint_dir)
    # Also saved the trained model


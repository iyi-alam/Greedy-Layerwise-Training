import torch
from tqdm import tqdm  #type: ignore
from torch import optim
import os
import numpy as np


def Accuracy(pred_labels, true_labels):
    pred_one_hot = torch.argmax(pred_labels, dim = 1)
    acc = torch.sum(pred_one_hot==true_labels)
    return acc

def train_step(model, train_loader, optimizer, criterion, accuracy, loss_history, save_steps, device, use_tqdm = True):
    # Model must be in train mode
    # model = model.to(device)
    epoch_loss = 0.0
    epoch_flops = 0
    acc = 0
    samples = 0
    if use_tqdm:
        train_itr = enumerate(tqdm(train_loader, desc = 'Processing training batches'))
    else:
        train_itr = enumerate(train_loader)

    for k, (images, labels) in train_itr:
        images, labels = images.to(device), labels.to(device)
        preds = model(images)

        loss = criterion(preds, labels)
        if np.isnan(loss.item()):
            print(preds)
            print(loss.item())
            return 'ERROR'

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        acc += accuracy(preds, labels).item()
        samples += len(images)

        # if (k+1)% save_steps == 0:
        #     loss_history['train'].append(epoch_loss/(k+1))

        # Compute flops
        batch_size, C, H, W = images.shape
        epoch_flops += batch_size * model.compute_flops(H,W)

    epoch_loss /= len(train_loader)
    acc /= samples
    return epoch_loss,acc, epoch_flops

def val_step(model, val_loader, criterion, accuracy, loss_history,save_steps, device, use_tqdm = True):
    epoch_loss = 0.0
    acc = 0
    samples = 0
    if use_tqdm:
        val_itr = enumerate(tqdm(val_loader, desc = 'Processing training batches'))
    else:
        val_itr = enumerate(val_loader)
    with torch.no_grad():
        for k,( images, labels) in val_itr:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)

            loss = criterion(preds, labels)
            epoch_loss += loss.item()
            acc += accuracy(preds, labels).item()
            samples += len(images)

            # if (k+1)% save_steps == 0:
            #     loss_history['val'].append(epoch_loss/(k+1))

    epoch_loss /= len(val_loader)
    acc /= samples

    return epoch_loss, acc

def test_step(model, test_loader, accuracy, device):
    acc = 0.0
    samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)

            acc += accuracy(preds, labels).item()
            samples += len(images)

    return acc/samples


def greedy_layerwise_training2(model,
                             train_dataloader,
                             test_dataloader, 
                             optimizer, 
                             criterion, 
                             initial_in_features, 
                             save_steps,
                             loss_history, 
                             acc_history, 
                             flops_history,
                             layer_ids_to_be_added,
                             add_layers,
                             update_optimizer,
                             device,
                             model_path,
                             training_params,
                             num_classes = 10,
                             last_epoch = -1,
                             best_acc = None,
                             use_tqdm = True,):
    
    model.to(device)

    MAX_EPOCHS = training_params['MAX_EPOCHS']
    EPOCHS_TO_ADD_LAYER = training_params['EPOCHS_TO_ADD_LAYER']
    MAX_EPOCHS_TO_ADD_LAYER = training_params['MAX_EPOCHS_TO_ADD_LAYER']
    LR = training_params['LR']
    MOMENTUM = training_params['MOMENTUM']
    FREEZE = training_params['FREEZE']

    curr_num_layers = 0
    total_flops = 0
    in_features = initial_in_features
    if best_acc is None:
        best_acc = 0.0


    for epoch in range(MAX_EPOCHS):

        # Add layer at correct epochs
        if (epoch > 0 and ((epoch+1) % EPOCHS_TO_ADD_LAYER == 0) and epoch < MAX_EPOCHS_TO_ADD_LAYER):
            if curr_num_layers >= len(layer_ids_to_be_added):
                # layer_id = layer_ids_to_be_added[-1]
                # curr_num_layers -= 1
                continue
            else:
                layer_id = layer_ids_to_be_added[curr_num_layers]

            if FREEZE:
                # disable gradient update for already trained layers
                for layer in model.layers:
                    for param in layer.parameters():
                        param.requires_grad = False

            if layer_id == 0:
                 add_layers(model, in_features, in_features, layer_id, num_classes, device)
                 #add_layers(model, in_features, in_features, layer_id, num_classes, device)
                 curr_num_layers += 1
            else:
                out_features = 2*in_features
                add_layers(model, in_features, out_features, layer_id, num_classes, device)
                curr_num_layers += 1
                in_features = out_features

            print('Added new layer. Current number of layers = ', curr_num_layers + 1)
            print('Last Added layer ID: ', layer_id)

            # Update optimizer
            optimizer = update_optimizer(optimizer, model, lr=LR, momentum=MOMENTUM)
            
        # Train step
        model.train()
        train_loss, train_acc, epoch_flops = train_step(model, train_dataloader, optimizer, criterion, Accuracy, loss_history, save_steps, device, use_tqdm= use_tqdm)
        loss_history['train'].append(train_loss)
        acc_history['train'].append(train_acc)
        flops_history.append(epoch_flops)
        total_flops += epoch_flops

        # Validation Step
        model.eval()
        val_loss, val_acc = val_step(model, test_dataloader, criterion, Accuracy, loss_history, save_steps, device, use_tqdm= use_tqdm)
        loss_history['val'].append(val_loss)
        acc_history['val'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}')
        

    print('\nTraining Finished!')

    train_metric = {
        'loss_history': loss_history,
        'acc_history': acc_history,
        'flops_history': flops_history,
        'last_epoch': max(epoch, last_epoch),
        'best_acc': best_acc
    }

    return train_metric


def greedy_layerwise_training(model,
                             train_dataloader,
                             test_dataloader, 
                             optimizer, 
                             criterion, 
                             initial_in_features, 
                             save_steps,
                             loss_history, 
                             acc_history, 
                             flops_history,
                             layer_ids_to_be_added,
                             add_layers,
                             update_optimizer,
                             device,
                             model_path,
                             metric_path,
                             training_params,
                             num_classes = 10,
                             last_epoch = -1,
                             best_acc = None,
                             use_tqdm = True,
                             use_scheduler = True):
    
    model.to(device)

    MAX_EPOCHS = training_params['MAX_EPOCHS']
    EPOCHS_TO_ADD_LAYER = training_params['EPOCHS_TO_ADD_LAYER']
    MAX_EPOCHS_TO_ADD_LAYER = training_params['MAX_EPOCHS_TO_ADD_LAYER']
    LR = training_params['LR']
    MOMENTUM = training_params['MOMENTUM']
    FREEZE = training_params['FREEZE']
    

    # Add warmup and scheduler parameters to training_params
    WARMUP_EPOCHS = training_params.get('WARMUP_EPOCHS', 5)  # Default to 5 epochs
    STEP_MILESTONES = training_params.get('STEP_MILESTONES', [30, 60])  # Default milestones

    # Initialize the step scheduler and warmup scheduler
    if use_scheduler:
        step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=STEP_MILESTONES, gamma=0.1)
        scheduler = WarmupLR(optimizer, warmup_epochs=WARMUP_EPOCHS, base_lr=LR, after_scheduler=step_scheduler)

    curr_num_layers = 0
    total_flops = 0
    in_features = initial_in_features
    if best_acc is None:
        best_acc = 0.0
    milestone = 0
    # model_load = True
    # if model_load and last_epoch != -1:
    #     model.load_state_dict(torch.load(model_path, weights_only= True))
    #     print('Loading the model at epoch: ', epoch)
    #     model_load = False


    for epoch in range(MAX_EPOCHS):

        if epoch == STEP_MILESTONES[milestone]:
            curr_lr = optimizer.param_groups[0]['lr']
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1*curr_lr
            milestone = min(milestone+1, len(STEP_MILESTONES)-1) 

        # Add layer at correct epochs
        if (epoch > 0 and ((epoch+1) % EPOCHS_TO_ADD_LAYER == 0) and epoch < MAX_EPOCHS_TO_ADD_LAYER):
            if curr_num_layers >= len(layer_ids_to_be_added):
                # layer_id = layer_ids_to_be_added[-1]
                # curr_num_layers -= 1
                continue
            else:
                layer_id = layer_ids_to_be_added[curr_num_layers]

            if FREEZE:
                # disable gradient update for already trained layers
                for layer in model.layers:
                    for param in layer.parameters():
                        param.requires_grad = False

            if layer_id == 0:
                 add_layers(model, in_features, in_features, layer_id, num_classes, device)
                 #add_layers(model, in_features, in_features, layer_id, num_classes, device)
                 curr_num_layers += 1
            else:
                out_features = 2*in_features
                add_layers(model, in_features, out_features, layer_id, num_classes, device)
                curr_num_layers += 1
                in_features = out_features

            print('Added new layer. Current number of layers = ', curr_num_layers + 1)
            print('Last Added layer ID: ', layer_id)

            # Update optimizer
            optimizer = update_optimizer(optimizer, model, lr=LR, momentum=MOMENTUM)
            
            # Reinitialize the scheduler with the new optimizer
            if use_scheduler:
                step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=STEP_MILESTONES, gamma=0.1)
                scheduler = WarmupLR(optimizer, warmup_epochs=WARMUP_EPOCHS, base_lr=LR, after_scheduler=step_scheduler)
                scheduler.current_epoch = epoch  # Preserve the current epoch count
            
        # Train step
        model.train()
        train_loss, train_acc, epoch_flops = train_step(model, train_dataloader, optimizer, criterion, Accuracy, loss_history, save_steps, device, use_tqdm= use_tqdm)
        loss_history['train'].append(train_loss)
        acc_history['train'].append(train_acc)
        flops_history.append(epoch_flops)
        total_flops += epoch_flops

        # Validation Step
        model.eval()
        val_loss, val_acc = val_step(model, test_dataloader, criterion, Accuracy, loss_history, save_steps, device, use_tqdm= use_tqdm)
        loss_history['val'].append(val_loss)
        acc_history['val'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}')

        if epoch % 10 == 0:
            train_metric = {
                'loss_history': loss_history,
                'acc_history': acc_history,
                'flops_history': flops_history,
                'last_epoch': max(epoch, last_epoch),
                'best_acc': best_acc
            }
            torch.save(train_metric, metric_path)
        
        #else:
        # Step the scheduler
        if use_scheduler:
            scheduler.step()
            #print('Skipping the Epoch...')
        

    print('\nTraining Finished!')

    train_metric = {
        'loss_history': loss_history,
        'acc_history': acc_history,
        'flops_history': flops_history,
        'last_epoch': max(epoch, last_epoch),
        'best_acc': best_acc
    }

    return train_metric

def greed_layerwise_training1(model,
                             train_dataloader,
                             test_dataloader, 
                             optimizer, 
                             criterion, 
                             initial_in_features, 
                             save_steps,
                             loss_history, 
                             acc_history, 
                             flops_history,
                             layer_ids_to_be_added,
                             add_layers,
                             update_optimizer,
                             device,
                             check_point,
                             model_path,
                             training_params):
    
    model.to(device)

    MAX_EPOCHS = training_params['MAX_EPOCHS']
    EPOCHS_TO_ADD_LAYER = training_params['EPOCHS_TO_ADD_LAYER']
    MAX_EPOCHS_TO_ADD_LAYER = training_params['MAX_EPOCHS_TO_ADD_LAYER']
    LR = training_params['LR']
    MOMENTUM = training_params['MOMENTUM']
    FREEZE = training_params['FREEZE']


    curr_num_layers = 0
    total_flops = 0
    in_features = initial_in_features
    for epoch in range(MAX_EPOCHS):

        # Add layer at correct epochs
        if (epoch > 0 and ((epoch+1) % EPOCHS_TO_ADD_LAYER == 0) and epoch < MAX_EPOCHS_TO_ADD_LAYER):
            layer_id = layer_ids_to_be_added[curr_num_layers]

            if FREEZE:
                # disable gradient update for already trained layers
                for layer in model.layers:
                    for param in layer.parameters():
                        param.requires_grad = False

            if layer_id == 0:
                 add_layers(model, in_features, in_features, layer_id, in_features, device)
                 add_layers(model, in_features, in_features, layer_id, in_features, device)
                 curr_num_layers += 2
            else:
                out_features = 2*in_features
                add_layers(model, in_features, out_features, layer_id, in_features, device)
                curr_num_layers += 1
                in_features = out_features

            print('Added new layer. Current number of layers = ', curr_num_layers + 1)
            print('Last Added layer ID: ', layer_id)

            # update optimizer
            optimizer = update_optimizer(optimizer, model, lr = LR, momentum = MOMENTUM)
             
        if epoch > check_point:

        # Train step
            model.train()
            train_loss, train_acc, epoch_flops = train_step(model, train_dataloader, optimizer, criterion, Accuracy, loss_history, save_steps, device)
            loss_history['train'].append(train_loss)
            acc_history['train'].append(train_acc)
            flops_history.append(epoch_flops)
            total_flops += epoch_flops

            # Validation Step
            model.eval()
            val_loss, val_acc = val_step(model, test_dataloader, criterion, Accuracy, loss_history, save_steps, device)
            loss_history['val'].append(val_loss)
            acc_history['val'].append(val_acc)


            print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:4f} | Val Acc: {val_acc:.4f}')

        else:
            print(f'Skipping {epoch+1} epoch to match checkpoint')

    print('\nTraining Finished!')
    torch.save(model.state_dict(), model_path)

    train_metric = {
        'loss_history': loss_history,
        'acc_history': acc_history,
        'flops_history': flops_history,
        'last_epoch': epoch
    }

    return train_metric

def e2e_training(model,
                train_dataloader,
                test_dataloader, 
                optimizer, 
                criterion, 
                initial_in_features, 
                save_steps,
                loss_history, 
                acc_history, 
                flops_history,
                layer_ids_to_be_added,
                device,
                training_params,
                use_tqdm = False,
                model_dir = './e2e_model.pth'):
    
    MAX_EPOCHS = training_params['MAX_EPOCHS']
    STEP_MILESTONES = training_params['STEP_MILESTONES']
    milestone = 0
    best_acc = 0.0
    
    model.to(device)
    total_flops = 0
    for epoch in range(MAX_EPOCHS):
        
        if epoch == STEP_MILESTONES[milestone]:
            curr_lr = optimizer.param_groups[0]['lr']
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1*curr_lr
            milestone = min(milestone+1, len(STEP_MILESTONES)-1)

        # Train step
        model.train()

        st = train_step(model, train_dataloader, optimizer, criterion, Accuracy, loss_history, save_steps, device, use_tqdm)
        if st == 'ERROR':
            break
        train_loss, train_acc, epoch_flops = st
        loss_history['train'].append(train_loss)
        acc_history['train'].append(train_acc)
        flops_history.append(epoch_flops)
        total_flops += epoch_flops

        # Validation Step
        model.eval()
        val_loss, val_acc = val_step(model, test_dataloader, criterion, Accuracy, loss_history, save_steps, device, use_tqdm)
        loss_history['val'].append(val_loss)
        acc_history['val'].append(val_acc)

        if val_acc > best_acc:
            val_acc = best_acc
            torch.save(model.state_dict(), model_dir)


        print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:4f} | Val Acc: {val_acc:.4f}')

    print('\nTraining Finished!')

    train_metric = {
        'loss_history': loss_history,
        'acc_history': acc_history,
        'flops_history': flops_history,
    }

    return train_metric


def imagenet_e2e_training(model,
                             train_dataloader,
                             test_dataloader, 
                             optimizer, 
                             criterion, 
                             initial_in_features, 
                             save_steps,
                             loss_history, 
                             acc_history, 
                             flops_history,
                             layer_ids_to_be_added,
                             add_layers,
                             update_optimizer,
                             device,
                             model_path,
                             metric_path,
                             training_params,
                             num_classes = 10,
                             last_epoch = -1,
                             best_acc = None,
                             use_tqdm = True,
                             use_scheduler = True):
    
    model.to(device)

    MAX_EPOCHS = training_params['MAX_EPOCHS']
    EPOCHS_TO_ADD_LAYER = training_params['EPOCHS_TO_ADD_LAYER']
    MAX_EPOCHS_TO_ADD_LAYER = training_params['MAX_EPOCHS_TO_ADD_LAYER']
    LR = training_params['LR']
    MOMENTUM = training_params['MOMENTUM']
    FREEZE = training_params['FREEZE']
    

    # Add warmup and scheduler parameters to training_params
    WARMUP_EPOCHS = training_params.get('WARMUP_EPOCHS', 5)  # Default to 5 epochs
    STEP_MILESTONES = training_params.get('STEP_MILESTONES', [30, 60])  # Default milestones

    # Initialize the step scheduler and warmup scheduler
    if use_scheduler:
        step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=STEP_MILESTONES, gamma=0.1)
        scheduler = WarmupLR(optimizer, warmup_epochs=WARMUP_EPOCHS, base_lr=LR, after_scheduler=step_scheduler)

    curr_num_layers = 0
    total_flops = 0
    in_features = initial_in_features
    if best_acc is None:
        best_acc = 0.0
    milestone = 0
    # model_load = True
    # if model_load and last_epoch != -1:
    #     model.load_state_dict(torch.load(model_path, weights_only= True))
    #     print('Loading the model at epoch: ', epoch)
    #     model_load = False

    continue_add_layers = True
    for epoch in range(MAX_EPOCHS):

        if epoch == STEP_MILESTONES[milestone]:
            curr_lr = optimizer.param_groups[0]['lr']
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.1*curr_lr
            milestone = min(milestone+1, len(STEP_MILESTONES)-1) 

        # Add layer at correct epochs
        if (epoch > 0 and epoch < MAX_EPOCHS_TO_ADD_LAYER and continue_add_layers):
            if curr_num_layers >= len(layer_ids_to_be_added):
                continue_add_layers = False
            else:
                layer_id = layer_ids_to_be_added[curr_num_layers]

            if FREEZE:
                # disable gradient update for already trained layers
                for layer in model.layers:
                    for param in layer.parameters():
                        param.requires_grad = False

            if layer_id == 0:
                 add_layers(model, in_features, in_features, layer_id, num_classes, device)
                 #add_layers(model, in_features, in_features, layer_id, num_classes, device)
                 curr_num_layers += 1
            else:
                out_features = 2*in_features
                add_layers(model, in_features, out_features, layer_id, num_classes, device)
                curr_num_layers += 1
                in_features = out_features

            print('Added new layer. Current number of layers = ', curr_num_layers + 1)
            print('Last Added layer ID: ', layer_id)

            # Update optimizer
            optimizer = update_optimizer(optimizer, model, lr=LR, momentum=MOMENTUM)
            
            # Reinitialize the scheduler with the new optimizer
            if use_scheduler:
                step_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=STEP_MILESTONES, gamma=0.1)
                scheduler = WarmupLR(optimizer, warmup_epochs=WARMUP_EPOCHS, base_lr=LR, after_scheduler=step_scheduler)
                scheduler.current_epoch = epoch  # Preserve the current epoch count
            
        # Train step
        model.train()
        train_loss, train_acc, epoch_flops = train_step(model, train_dataloader, optimizer, criterion, Accuracy, loss_history, save_steps, device, use_tqdm= use_tqdm)
        loss_history['train'].append(train_loss)
        acc_history['train'].append(train_acc)
        flops_history.append(epoch_flops)
        total_flops += epoch_flops

        # Validation Step
        model.eval()
        val_loss, val_acc = val_step(model, test_dataloader, criterion, Accuracy, loss_history, save_steps, device, use_tqdm= use_tqdm)
        loss_history['val'].append(val_loss)
        acc_history['val'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}')

        if epoch % 10 == 0:
            train_metric = {
                'loss_history': loss_history,
                'acc_history': acc_history,
                'flops_history': flops_history,
                'last_epoch': max(epoch, last_epoch),
                'best_acc': best_acc
            }
            torch.save(train_metric, metric_path)
        
        #else:
        # Step the scheduler
        if use_scheduler:
            scheduler.step()
            #print('Skipping the Epoch...')
        

    print('\nTraining Finished!')

    train_metric = {
        'loss_history': loss_history,
        'acc_history': acc_history,
        'flops_history': flops_history,
        'last_epoch': max(epoch, last_epoch),
        'best_acc': best_acc
    }

    return train_metric



def static_training(model,
                train_dataloader,
                test_dataloader, 
                optimizer, 
                criterion, 
                initial_in_features, 
                save_steps,loss_history, 
                acc_history, 
                flops_history,
                device,
                max_epochs,
                model_path,
                best_acc):
    
    model.to(device)

    # Get current best accuracy:
    
    for epoch in range(max_epochs):

        # Train step
        model.train()
        train_loss, train_acc, epoch_flops = train_step(model, train_dataloader, optimizer, criterion, Accuracy, loss_history, save_steps, device)
        loss_history['train'].append(train_loss)
        acc_history['train'].append(train_acc)
        flops_history.append(epoch_flops)

        # Validation Step
        model.eval()
        val_loss, val_acc = val_step(model, test_dataloader, criterion, Accuracy, loss_history, save_steps,device)
        loss_history['val'].append(val_loss)
        acc_history['val'].append(val_acc)

        if (val_acc > best_acc):
            best_acc = val_acc
            model_state_dict = model.state_dict()
            torch.save(model_state_dict, model_path)

        print(f'Epoch: {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:4f} | Val Acc: {val_acc:.4f}')

    print('\nTraining Finished!')

    train_metric = {
        'loss_history': loss_history,
        'acc_history': acc_history,
        'flops_history': flops_history,
    }

    return train_metric

class WarmupLR:
    """
    Custom learning rate scheduler with warmup.
    
    Args:
        optimizer: The optimizer to adjust the learning rate for
        warmup_epochs: Number of epochs for warmup
        base_lr: Target learning rate after warmup
        after_scheduler: Scheduler to use after warmup (e.g., MultiStepLR)
    """
    def __init__(self, optimizer, warmup_epochs, base_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.after_scheduler = after_scheduler
        self.current_epoch = 0
        
        # Store initial learning rates for each param group
        self.init_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
    
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr_scale = self.current_epoch / self.warmup_epochs
            for param_group, init_lr in zip(self.optimizer.param_groups, self.init_lrs):
                param_group['lr'] = lr_scale * self.base_lr
        else:
            # After warmup, use the provided scheduler
            if self.after_scheduler:
                self.after_scheduler.step()
            # Ensure the learning rate doesn't fall below a minimum
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], 1e-6)
    
    def get_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
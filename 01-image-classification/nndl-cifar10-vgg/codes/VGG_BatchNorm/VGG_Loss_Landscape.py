import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
import multiprocessing
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Create directories if they don't exist
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# Make sure you are using the right device.
device = torch.device("cuda" if torch.cuda.is_available() else \
                      "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else \
                      "cpu")

# This function is used to calculate the accuracy of model classification
def get_accuracy(model, data_loader, device):
    """
    Calculate accuracy of model predictions on the given dataset
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    """
    Train model and record losses and gradients for visualization
    """
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            
            # Record loss values for landscape visualization
            loss_list.append(loss.item())
            learning_curve[epoch] += loss.item()
            
            loss.backward()
            
            # Record gradient information
            if hasattr(model, 'classifier') and len(list(model.classifier.children())) >= 5:
                # For VGG models
                gradient_layer = 4 if len(list(model.classifier.children())) > 4 else -2
                grad.append(model.classifier[gradient_layer].weight.grad.norm().item())
            
            optimizer.step()

        losses_list.append(loss_list)
        grads.append(grad)
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))

        learning_curve[epoch] /= batches_n
        axes[0].plot(learning_curve)
        axes[0].set_title('Training Loss')

        # Test your model
        model.eval()
        train_accuracy = get_accuracy(model, train_loader, device)
        val_accuracy = get_accuracy(model, val_loader, device)
        train_accuracy_curve[epoch] = train_accuracy
        val_accuracy_curve[epoch] = val_accuracy
        
        axes[1].plot(train_accuracy_curve, label='Train')
        axes[1].plot(val_accuracy_curve, label='Validation')
        axes[1].legend()
        axes[1].set_title('Accuracy')
        
        plt.savefig(os.path.join(figures_path, f'{model.__class__.__name__}_training_curve.png'))
        plt.close()
        
        # Save best model
        if val_accuracy > max_val_accuracy:
            max_val_accuracy = val_accuracy
            max_val_accuracy_epoch = epoch
            if best_model_path:
                torch.save(model.state_dict(), best_model_path)
                
        print(f'Epoch {epoch+1}/{epochs_n}, Loss: {learning_curve[epoch]:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')

    print(f'Best validation accuracy: {max_val_accuracy:.2f}% at epoch {max_val_accuracy_epoch+1}')
    return losses_list, grads

# Generate loss landscape plots for each model type
def plot_loss_landscape(model_name, model_losses):
    """
    Plot loss landscape visualization showing min/max losses across different learning rates
    """
    plt.figure(figsize=(10, 6))
    
    # For each training step
    num_steps = min(len(model_losses[lr][0]) for lr in model_losses)
    steps = range(1, num_steps + 1)
    
    min_curve = []
    max_curve = []
    
    # For each step, find min and max loss across all learning rates
    for step in range(num_steps):
        step_losses = []
        for lr in model_losses:
            if step < len(model_losses[lr][0]):
                step_losses.append(model_losses[lr][0][step])
        
        min_curve.append(min(step_losses))
        max_curve.append(max(step_losses))
    
    plt.fill_between(steps, min_curve, max_curve, alpha=0.3, label='Loss range')
    plt.plot(steps, min_curve, 'b--', label='Min loss')
    plt.plot(steps, max_curve, 'r--', label='Max loss')
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title(f'Loss Landscape for {model_name}')
    plt.legend()
    plt.savefig(os.path.join(figures_path, f'{model_name}_loss_landscape.png'))
    plt.close()

# Compare VGG_A and VGG_A_BatchNorm
def compare_models_performance():
    """
    Plot comparative metrics for VGG_A and VGG_A_BatchNorm
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Compare average training curves
    for model_name, loss_data in all_losses.items():
        avg_losses = []
        for epoch in range(epo):
            epoch_losses = []
            for lr, losses in loss_data.items():
                # Average loss for this epoch across batches
                if len(losses) > epoch:
                    epoch_losses.append(sum(losses[epoch]) / len(losses[epoch]))
            
            if epoch_losses:
                avg_losses.append(sum(epoch_losses) / len(epoch_losses))
        
        axes[0].plot(range(1, len(avg_losses) + 1), avg_losses, label=model_name)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Average Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    
    # Compare gradient stability
    for model_name, grad_data in all_grads.items():
        std_grads = []
        for epoch in range(epo):
            epoch_grads = []
            for lr, grads in grad_data.items():
                if len(grads) > epoch and len(grads[epoch]) > 0:
                    epoch_grads.extend(grads[epoch])
            
            if epoch_grads:
                std_grads.append(np.std(epoch_grads))
        
        axes[1].plot(range(1, len(std_grads) + 1), std_grads, label=model_name)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Gradient Standard Deviation')
    axes[1].set_title('Gradient Stability Comparison')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'model_comparison.png'))
    plt.close()

if __name__ == '__main__':
    # 添加多进程支持
    multiprocessing.freeze_support()
    
    # 打印设备信息
    print(f"Current device: {device}")
    if device.type == 'mps':
        print("✅ Metal Performance Shaders (MPS) is activated")
    elif device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Running on CPU")

    # Initialize your data loader
    train_loader = get_cifar_loader(train=True, num_workers=num_workers)
    val_loader = get_cifar_loader(train=False, num_workers=num_workers)
    for X, y in train_loader:
        print(f"Input batch shape: {X.shape}")
        print(f"Target batch shape: {y.shape}")
        print(f"Input dtype: {X.dtype}, Target dtype: {y.dtype}")
        break

    # Create directories for saving results
    os.makedirs('results', exist_ok=True)
    loss_save_path = 'results'
    grad_save_path = 'results'

    # Train models with various learning rates
    learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]
    model_types = [VGG_A, VGG_A_BatchNorm]
    all_losses = {model_type.__name__: {} for model_type in model_types}
    all_grads = {model_type.__name__: {} for model_type in model_types}

    epo = 20

    # Loop through models and learning rates
    for model_class in model_types:
        for lr in learning_rates:
            print(f"\nTraining {model_class.__name__} with learning rate: {lr}")
            
            # Set random seeds for reproducibility
            set_random_seeds(seed_value=2020, device=device)
            
            # Create model
            model = model_class()
            
            # Setup optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            # Train model
            best_model_path = os.path.join(models_path, f"{model_class.__name__}_lr{lr}.pth")
            losses, gradients = train(model, optimizer, criterion, train_loader, val_loader, 
                                    epochs_n=epo, best_model_path=best_model_path)
            
            # Save results
            all_losses[model_class.__name__][lr] = losses
            all_grads[model_class.__name__][lr] = gradients
            
            # Save raw data
            np.save(os.path.join(loss_save_path, f'{model_class.__name__}_lr{lr}_loss.npy'), losses)
            np.save(os.path.join(grad_save_path, f'{model_class.__name__}_lr{lr}_grads.npy'), gradients)

    # Generate landscapes for each model type
    for model_name, loss_data in all_losses.items():
        plot_loss_landscape(model_name, loss_data)

    # Generate comparative plots
    compare_models_performance()

    print("Training complete. Results saved in 'results' directory.")
    print("Visualizations saved in reports/figures directory.")

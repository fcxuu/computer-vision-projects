"""
Train CIFAR-10 classification network with different architectures and optimization strategies
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import multiprocessing

from VGG_BatchNorm.models.vgg import VGG_A, VGG_A_BatchNorm, VGG_A_Dropout, VGG_A_Light
from VGG_BatchNorm.data.loaders import get_cifar_loader

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else \
                     "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else \
                     "cpu")
# print(f"Using device: {device}")

# Create directories for saving results
os.makedirs('results', exist_ok=True)
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

def visualize_filters(model, layer_idx=0, single_channel=True, figsize=(8, 8)):
    """
    Visualize filters from the first convolutional layer of the model
    
    Args:
        model: PyTorch model
        layer_idx: Index of the convolutional layer to visualize
        single_channel: If True, visualize single channels of each filter
        figsize: Figure size
    """
    # Get the first convolutional layer
    if hasattr(model, 'features'):
        # For VGG-style models
        layers = list(model.features.children())
    else:
        # For stage-based models
        layers = list(model.stage1.children())
    
    conv_layer = None
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            if layer_idx == 0:
                conv_layer = layer
                break
            layer_idx -= 1
    
    if conv_layer is None:
        print("No convolutional layer found!")
        return
    
    # Get filters
    weights = conv_layer.weight.data.cpu()
    num_filters = weights.shape[0]
    
    # Create figure to display filters
    num_cols = 8
    num_rows = (num_filters + num_cols - 1) // num_cols  # Ceiling division
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(num_filters):
        if i < len(axes):
            filter_weights = weights[i]
            if single_channel and filter_weights.shape[0] > 1:
                # For RGB filters, just take the first channel
                filter_display = filter_weights[0]
            else:
                # For grayscale or if we want to average RGB channels
                filter_display = filter_weights.mean(dim=0)
            
            # Normalize for better visualization
            filter_display = (filter_display - filter_display.min()) / (filter_display.max() - filter_display.min() + 1e-8)
            
            # Display the filter
            axes[i].imshow(filter_display, cmap='viridis')
            axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(num_filters, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'results/figures/{model.__class__.__name__}_filters.png')
    plt.close()
    print(f"Filter visualization saved for {model.__class__.__name__}")

def train_model(model_name, optimizer_name='adam', learning_rate=0.001, batch_size=128, 
                num_epochs=30, use_scheduler=True, weight_decay=0, num_workers=0):
    """
    Train a model with the specified parameters
    
    Args:
        model_name: Name of the model to train ('vgg_a', 'vgg_a_bn', 'vgg_a_dropout')
        optimizer_name: Name of the optimizer to use ('adam', 'sgd', 'rmsprop')
        learning_rate: Learning rate
        batch_size: Batch size
        num_epochs: Number of epochs
        use_scheduler: Whether to use a learning rate scheduler
        weight_decay: L2 regularization parameter
        num_workers: Number of workers for data loading
        
    Returns:
        trained model and training history
    """
    # Create model
    if model_name == 'vgg_a':
        model = VGG_A()
    elif model_name == 'vgg_a_bn':
        model = VGG_A_BatchNorm()
    elif model_name == 'vgg_a_dropout':
        model = VGG_A_Dropout()
    elif model_name == 'vgg_a_light':
        model = VGG_A_Light()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model = model.to(device)
    
    # Get data loaders
    train_loader = get_cifar_loader(train=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = get_cifar_loader(train=False, batch_size=batch_size, num_workers=num_workers)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Define scheduler
    scheduler = None
    if use_scheduler:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Train model
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Backward + optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        if scheduler is not None:
            scheduler.step()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.float() / len(train_loader.dataset) * 100
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.cpu().item())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.float() / len(val_loader.dataset) * 100
        
        history['val_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc.cpu().item())
        
        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
        
        # Save best model
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            # Save model weights
            torch.save(model.state_dict(), f'results/models/{model_name}_{optimizer_name}_lr{learning_rate}.pth')
    
    # Create training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/figures/{model_name}_{optimizer_name}_lr{learning_rate}.png')
    plt.close()
    
    # Visualize model filters
    visualize_filters(model)
    
    return model, history

def main():
    """
    Main function to run the training with command-line arguments
    """
    # 打印设备信息
    print(f"Using device: {device}")
    if device.type == 'mps':
        print("✅ Metal Performance Shaders (MPS) is activated")
    elif device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Running on CPU")
        
    parser = argparse.ArgumentParser(description='Train CIFAR-10 classification models')
    parser.add_argument('--model', type=str, default='vgg_a', 
                        choices=['vgg_a', 'vgg_a_bn', 'vgg_a_dropout', 'vgg_a_light'],
                        help='Model architecture to use')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd', 'rmsprop'],
                        help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
    parser.add_argument('--no_scheduler', action='store_true', help='Disable learning rate scheduler')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Number of data loading workers (0 for no multiprocessing)')
    
    args = parser.parse_args()
    
    # Train model with specified parameters
    model, history = train_model(
        model_name=args.model,
        optimizer_name=args.optimizer,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        use_scheduler=not args.no_scheduler,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers
    )
    
    print(f"Training complete for {args.model} with {args.optimizer} optimizer")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")

if __name__ == "__main__":
    # 添加多进程支持
    multiprocessing.freeze_support()
    main() 
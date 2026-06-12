import numpy as np
import matplotlib.pyplot as plt

def visualize_mlp_weights(model, layer_idx=0, figsize=(10, 10), nrows=None, ncols=None, cmap='viridis'):
    """
    Visualize weights of a linear layer in an MLP model
    
    Args:
        model: MLP model
        layer_idx: index of the linear layer to visualize (0-indexed among Linear layers only)
        figsize: figure size
        nrows, ncols: number of rows and columns for the grid plot
        cmap: colormap for visualization
    """
    # Get all Linear layers
    linear_layers = [layer for layer in model.layers if hasattr(layer, 'W')]
    
    if layer_idx >= len(linear_layers):
        raise ValueError(f"Layer index {layer_idx} out of range. Model has {len(linear_layers)} Linear layers.")
    
    # Get the weights of the specified layer
    weights = linear_layers[layer_idx].W
    n_input, n_output = weights.shape
    
    # Determine if the weights can be reshaped to a square for visualization
    is_image_input = np.sqrt(n_input).is_integer()
    
    if is_image_input:
        # Reshape weights to visualize as images (for first layer)
        side_length = int(np.sqrt(n_input))
        weights_reshaped = weights.reshape(side_length, side_length, n_output)
        
        # Set up the grid for plotting
        if nrows is None or ncols is None:
            ncols = int(np.ceil(np.sqrt(n_output)))
            nrows = int(np.ceil(n_output / ncols))
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        fig.suptitle(f'Layer {layer_idx} Weights Visualization', fontsize=16)
        
        # Plot each neuron's weights as an image
        for i in range(nrows):
            for j in range(ncols):
                idx = i * ncols + j
                if idx < n_output:
                    if nrows > 1 and ncols > 1:
                        ax = axes[i, j]
                    elif nrows > 1:
                        ax = axes[i]
                    elif ncols > 1:
                        ax = axes[j]
                    else:
                        ax = axes
                    
                    # Plot weight image
                    im = ax.imshow(weights_reshaped[:, :, idx], cmap=cmap)
                    ax.set_title(f'Neuron {idx}')
                    ax.axis('off')
                else:
                    if nrows > 1 and ncols > 1:
                        axes[i, j].axis('off')
                    elif nrows > 1:
                        axes[i].axis('off')
                    elif ncols > 1:
                        axes[j].axis('off')
        
        plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    else:
        # For non-image weights, plot as a heatmap
        plt.figure(figsize=figsize)
        plt.imshow(weights, cmap=cmap, aspect='auto')
        plt.colorbar()
        plt.title(f'Layer {layer_idx} Weights Heatmap ({n_input} x {n_output})')
        plt.xlabel('Output Neurons')
        plt.ylabel('Input Features')
    
    plt.tight_layout()
    return plt.gcf()

def visualize_cnn_kernels(model, layer_idx=0, figsize=(10, 10), nrows=None, ncols=None, cmap='viridis'):
    """
    Visualize convolutional kernels of a CNN model
    
    Args:
        model: CNN model
        layer_idx: index of the conv layer to visualize (0-indexed among Conv layers only)
        figsize: figure size
        nrows, ncols: number of rows and columns for the grid plot
        cmap: colormap for visualization
    """
    # Get all Conv layers
    conv_layers = [layer for layer in model.layers if hasattr(layer, 'kernel_size')]
    
    if layer_idx >= len(conv_layers):
        raise ValueError(f"Layer index {layer_idx} out of range. Model has {len(conv_layers)} Conv layers.")
    
    # Get the weights of the specified layer
    conv_layer = conv_layers[layer_idx]
    # Weights shape: [out_channels, in_channels, kernel_height, kernel_width]
    weights = conv_layer.W
    out_channels, in_channels, k_h, k_w = weights.shape
    
    # Set up the grid for plotting
    if nrows is None or ncols is None:
        ncols = in_channels
        nrows = out_channels
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(f'Conv Layer {layer_idx} Kernels Visualization', fontsize=16)
    
    # Plot each kernel
    for out_idx in range(out_channels):
        for in_idx in range(in_channels):
            if nrows > 1 and ncols > 1:
                ax = axes[out_idx, in_idx]
            elif nrows > 1:
                ax = axes[out_idx]
            elif ncols > 1:
                ax = axes[in_idx]
            else:
                ax = axes
            
            # Plot kernel
            im = ax.imshow(weights[out_idx, in_idx], cmap=cmap)
            ax.set_title(f'Out:{out_idx}, In:{in_idx}')
            ax.axis('off')
    
    plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return plt.gcf()

def visualize_model_training(runner, figsize=(12, 5)):
    """
    Visualize training and validation metrics
    
    Args:
        runner: the RunnerM instance after training
        figsize: figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot training and validation loss
    axes[0].plot(runner.train_loss, label='Training Loss')
    axes[0].plot(runner.dev_loss, label='Validation Loss')
    axes[0].set_title('Loss vs. Iterations')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Plot training and validation accuracy
    axes[1].plot(runner.train_scores, label='Training Accuracy')
    axes[1].plot(runner.dev_scores, label='Validation Accuracy')
    axes[1].set_title('Accuracy vs. Iterations')
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    return fig 
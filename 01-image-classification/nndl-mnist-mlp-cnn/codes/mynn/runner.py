import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):
        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")
        eval_every = kwargs.get("eval_every", 100)  # Evaluation frequency

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0
        total_iterations = 0

        for epoch in range(num_epochs):
            X, y = train_set

            assert X.shape[0] == y.shape[0]

            idx = np.random.permutation(range(X.shape[0]))

            X = X[idx]
            y = y[idx]

            num_batches = int(X.shape[0] / self.batch_size) + 1

            for iteration in range(num_batches):
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Only evaluate on validation set at specific iterations
                if total_iterations % eval_every == 0:
                    dev_score, dev_loss = self.evaluate(dev_set)
                    self.dev_scores.append(dev_score)
                    self.dev_loss.append(dev_loss)
                    
                    # Update best model
                    if dev_score > best_score:
                        save_path = os.path.join(save_dir, 'best_model.pickle')
                        self.save_model(save_path)
                        print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                        best_score = dev_score
                else:
                    # If not evaluating, keep previous value
                    if self.dev_scores:
                        self.dev_scores.append(self.dev_scores[-1])
                        self.dev_loss.append(self.dev_loss[-1])
                    else:
                        # First iteration needs initial evaluation
                        dev_score, dev_loss = self.evaluate(dev_set)
                        self.dev_scores.append(dev_score)
                        self.dev_loss.append(dev_loss)

                if (iteration) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score}")
                    print(f"[Dev] loss: {self.dev_loss[-1]}, score: {self.dev_scores[-1]}")
                    
                total_iterations += 1

        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss
    
    def save_model(self, save_path):
        self.model.save_model(save_path)
        
    def visualize_training(self, figsize=(12, 5)):
        """
        Visualize training and validation metrics
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot training and validation loss
        axes[0].plot(self.train_loss, label='Training Loss')
        axes[0].plot(self.dev_loss, label='Validation Loss')
        axes[0].set_title('Loss vs. Iterations')
        axes[0].set_xlabel('Iterations')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # Plot training and validation accuracy
        axes[1].plot(self.train_scores, label='Training Accuracy')
        axes[1].plot(self.dev_scores, label='Validation Accuracy')
        axes[1].set_title('Accuracy vs. Iterations')
        axes[1].set_xlabel('Iterations')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        
        plt.tight_layout()
        return fig

def visualize_mlp_weights(model, layer_idx=0, figsize=(10, 10), nrows=None, ncols=None, cmap='viridis'):
    """
    Visualize weights of a linear layer in an MLP model
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
from abc import abstractmethod
import numpy as np
from scipy.ndimage import rotate, shift, zoom
import random

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        # Save input for backward pass
        self.input = X
        # Linear transformation: Y = X * W + b
        return np.dot(X, self.W) + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        # Compute gradients for parameters
        # dL/dW = X^T * dL/dY
        self.grads['W'] = np.dot(self.input.T, grad)
        
        # dL/db = sum(dL/dY, axis=0)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        
        # Add weight decay gradient if enabled
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        
        # Compute gradient for input
        # dL/dX = dL/dY * W^T
        return np.dot(grad, self.W.T)
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias
        # W shape: [out_channels, in_channels, kernel_height, kernel_width]
        self.W = initialize_method(size=(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.b = initialize_method(size=(out_channels,))
        
        self.grads = {'W': None, 'b': None}
        self.params = {'W': self.W, 'b': self.b}
        self.input = None  # Store input for backward pass
        
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out_channels, in_channels, kernel_h, kernel_w]
        """
        self.input = X
        batch_size, in_channels, in_height, in_width = X.shape
        
        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size[1]) // self.stride + 1
        
        # Initialize output tensor
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Apply padding if needed
        if self.padding > 0:
            padded_X = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        else:
            padded_X = X
        
        # Perform convolution
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        w_start = w_out * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        
                        # Extract the region to perform convolution on
                        X_region = padded_X[b, :, h_start:h_end, w_start:w_end]
                        
                        # Perform convolution and add bias
                        output[b, c_out, h_out, w_out] = np.sum(X_region * self.W[c_out]) + self.b[c_out]
        
        return output

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, out_height, out_width]
        """
        batch_size, _, out_height, out_width = grads.shape
        _, in_channels, in_height, in_width = self.input.shape
        
        # Initialize gradients for parameters and input
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(self.input)
        
        # If padding was applied in forward, apply it to input for backward
        if self.padding > 0:
            padded_X = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
            # Also initialize a padded version of dX
            padded_dX = np.zeros((batch_size, in_channels, in_height + 2 * self.padding, in_width + 2 * self.padding))
        else:
            padded_X = self.input
            padded_dX = dX
        
        # Compute gradients
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        w_start = w_out * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        
                        # Gradient of bias
                        db[c_out] += grads[b, c_out, h_out, w_out]
                        
                        # Gradient of weights
                        dW[c_out] += padded_X[b, :, h_start:h_end, w_start:w_end] * grads[b, c_out, h_out, w_out]
                        
                        # Gradient of input
                        padded_dX[b, :, h_start:h_end, w_start:w_end] += self.W[c_out] * grads[b, c_out, h_out, w_out]
        
        # Apply weight decay if enabled
        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W
        
        # Store gradients
        self.grads['W'] = dW
        self.grads['b'] = db
        
        # Extract the gradient of the input (without padding)
        if self.padding > 0:
            dX = padded_dX[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = padded_dX
        
        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class Sigmoid(Layer):
    """
    Sigmoid activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.output = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # Sigmoid: 1 / (1 + e^(-x))
        X_clipped = np.clip(X, -500, 500)
        self.output = 1 / (1 + np.exp(-X_clipped))
        return self.output
    
    def backward(self, grads):
        # Gradient of sigmoid: σ(x) * (1 - σ(x))
        return grads * self.output * (1 - self.output)

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.optimizable = False
        self.has_softmax = True  # By default, softmax is included
        self.predictions = None
        self.labels = None
        self.batch_size = None
        self.grads = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.batch_size = predicts.shape[0]
        self.labels = labels
        
        # Apply softmax if required
        if self.has_softmax:
            self.predictions = softmax(predicts)
        else:
            self.predictions = predicts
            
        # Create one-hot encoding of labels
        one_hot_labels = np.zeros((self.batch_size, self.max_classes))
        one_hot_labels[np.arange(self.batch_size), labels] = 1
        
        # Calculate cross-entropy loss
        # Adding small epsilon to avoid log(0)
        epsilon = 1e-10
        log_likelihood = -np.log(self.predictions + epsilon) * one_hot_labels
        loss = np.sum(log_likelihood) / self.batch_size
        
        return loss
    
    def backward(self):
        # Compute gradients from the loss to the input
        # For softmax + cross-entropy, the gradient is (softmax_output - one_hot_labels)
        one_hot_labels = np.zeros((self.batch_size, self.max_classes))
        one_hot_labels[np.arange(self.batch_size), self.labels] = 1
        
        if self.has_softmax:
            # When softmax is included, the gradient simplifies to (predictions - one_hot_labels)
            self.grads = (self.predictions - one_hot_labels) / self.batch_size
        else:
            # If no softmax is applied, need to compute cross-entropy gradients directly
            # This branch is generally not used as softmax is typically applied
            self.grads = -one_hot_labels / (self.predictions + 1e-10) / self.batch_size
        
        # Send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self

class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

# Data augmentation functions
def augment_image(image, label, rotate_range=15, shift_range=2, zoom_range=(0.9, 1.1)):
    """
    Apply random transformations to image for data augmentation
    """
    # Make a copy of the image
    img = image.copy()
    
    # Random rotation
    if rotate_range > 0:
        angle = random.uniform(-rotate_range, rotate_range)
        img = rotate(img, angle, reshape=False, mode='nearest')
    
    # Random shift
    if shift_range > 0:
        dx = random.uniform(-shift_range, shift_range)
        dy = random.uniform(-shift_range, shift_range)
        img = shift(img, [dy, dx], mode='nearest')
    
    # Random zoom
    if zoom_range[0] < zoom_range[1]:
        zoom_factor = random.uniform(zoom_range[0], zoom_range[1])
        shape = img.shape
        img = zoom(img, zoom_factor, mode='nearest')
        
        # Resize back to original size
        if zoom_factor > 1:  # If zoomed in, crop center
            new_shape = img.shape
            start_y = (new_shape[0] - shape[0]) // 2
            start_x = (new_shape[1] - shape[1]) // 2
            img = img[start_y:start_y+shape[0], start_x:start_x+shape[1]]
        else:  # If zoomed out, pad with zeros
            new_shape = img.shape
            pad_y = (shape[0] - new_shape[0]) // 2
            pad_x = (shape[1] - new_shape[1]) // 2
            new_img = np.zeros(shape)
            new_img[pad_y:pad_y+new_shape[0], pad_x:pad_x+new_shape[1]] = img
            img = new_img
    
    return img, label

def augment_dataset(images, labels, augmentation_factor=1):
    """
    Augment dataset by creating additional samples
    """
    # Handle different image formats
    if len(images.shape) == 4:  # [batch, channels, height, width]
        N, C, H, W = images.shape
        images_reshaped = images.transpose(0, 2, 3, 1).reshape(N, H, W, C)
        is_4d = True
    else:  # [batch, height*width]
        N = images.shape[0]
        H = W = int(np.sqrt(images.shape[1]))
        images_reshaped = images.reshape(N, H, W)
        is_4d = False
    
    augmented_images = [images]
    augmented_labels = [labels]
    
    # Create augmented samples
    for i in range(augmentation_factor):
        aug_batch_images = []
        for j in range(N):
            if is_4d:
                # For CNN input format
                aug_img = np.zeros_like(images_reshaped[j])
                for c in range(C):
                    aug_img[:,:,c], _ = augment_image(images_reshaped[j,:,:,c], labels[j])
                aug_batch_images.append(aug_img.transpose(2, 0, 1))
            else:
                # For MLP input format
                aug_img, _ = augment_image(images_reshaped[j], labels[j])
                aug_batch_images.append(aug_img.flatten())
        
        augmented_images.append(np.array(aug_batch_images))
        augmented_labels.append(labels)
    
    # Combine all augmented data
    augmented_images = np.vstack(augmented_images)
    augmented_labels = np.hstack(augmented_labels)
    
    return augmented_images, augmented_labels
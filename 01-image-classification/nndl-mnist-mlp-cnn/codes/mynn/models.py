from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    if act_func == 'Logistic':
                        layer_f = Sigmoid()
                    elif act_func == 'ReLU':
                        layer_f = ReLU()
                    else:
                        raise ValueError(f"Unsupported activation function: {act_func}")
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+2]['lambda']
            if self.act_func == 'Logistic':
                layer_f = Sigmoid()
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)

    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, channels_list=None, kernel_sizes=None, act_func=None, lambda_list=None, fc_sizes=None, input_shape=None):
        super().__init__()
        self.channels_list = channels_list  # [in_channels, conv1_out, conv2_out, ...]
        self.kernel_sizes = kernel_sizes    # [kernel_size1, kernel_size2, ...]
        self.act_func = act_func
        self.lambda_list = lambda_list      # Weight decay parameters
        self.fc_sizes = fc_sizes            # Fully connected layer sizes after conv layers
        self.input_shape = input_shape      # [batch_size, channels, height, width]
        self.layers = []
        
        if channels_list is not None and kernel_sizes is not None and act_func is not None and input_shape is not None:
            # Initialize convolutional layers
            h, w = input_shape[2], input_shape[3]
            
            for i in range(len(channels_list) - 1):
                # Add convolutional layer
                conv_layer = conv2D(in_channels=channels_list[i], 
                                   out_channels=channels_list[i + 1], 
                                   kernel_size=kernel_sizes[i])
                if lambda_list is not None:
                    conv_layer.weight_decay = True
                    conv_layer.weight_decay_lambda = lambda_list[i]
                
                self.layers.append(conv_layer)
                
                # Update dimensions for next layer
                h = h - kernel_sizes[i] + 1
                w = w - kernel_sizes[i] + 1
                
                # Add activation function
                if act_func == 'Logistic':
                    self.layers.append(Sigmoid())
                elif act_func == 'ReLU':
                    self.layers.append(ReLU())
                else:
                    raise ValueError(f"Unsupported activation function: {act_func}")
            
            # Initialize fully connected layers after convolutional layers
            if fc_sizes is not None:
                # Calculate input size for the first FC layer
                fc_in_size = channels_list[-1] * h * w
                fc_sizes = [fc_in_size] + fc_sizes
                
                for i in range(len(fc_sizes) - 1):
                    # Add fully connected layer
                    fc_layer = Linear(in_dim=fc_sizes[i], out_dim=fc_sizes[i + 1])
                    
                    if lambda_list is not None and i + len(channels_list) - 1 < len(lambda_list):
                        fc_layer.weight_decay = True
                        fc_layer.weight_decay_lambda = lambda_list[i + len(channels_list) - 1]
                    
                    self.layers.append(fc_layer)
                    
                    # Add activation function for all but the last fully connected layer
                    if i < len(fc_sizes) - 2:
                        if act_func == 'Logistic':
                            self.layers.append(Sigmoid())
                        elif act_func == 'ReLU':
                            self.layers.append(ReLU())

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        Forward pass through the CNN model
        X: input tensor of shape [batch_size, channels, height, width]
        """
        assert self.channels_list is not None and self.kernel_sizes is not None, \
            'Model has not been initialized yet'
        
        outputs = X
        conv_output = None
        
        for i, layer in enumerate(self.layers):
            # If transitioning from conv to fc layers, reshape the output
            if isinstance(layer, Linear) and (isinstance(self.layers[i-1], conv2D) or 
                                              isinstance(self.layers[i-1], ReLU) and isinstance(self.layers[i-2], conv2D) or
                                              isinstance(self.layers[i-1], Sigmoid) and isinstance(self.layers[i-2], conv2D)):
                # Flatten conv output for FC layer
                batch_size = outputs.shape[0]
                outputs = outputs.reshape(batch_size, -1)
            
            outputs = layer(outputs)
            
        return outputs

    def backward(self, loss_grad):
        """
        Backward pass through the CNN model
        loss_grad: gradient from the loss function
        """
        grads = loss_grad
        
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
            
        return grads
    
    def load_model(self, param_list):
        """
        Load model parameters from a file
        param_list: path to the pickle file containing model parameters
        """
        with open(param_list, 'rb') as f:
            params = pickle.load(f)
        
        self.channels_list = params[0]
        self.kernel_sizes = params[1]
        self.act_func = params[2]
        self.fc_sizes = params[3]
        self.input_shape = params[4]
        
        # Recreate the model architecture
        self.__init__(self.channels_list, self.kernel_sizes, self.act_func, 
                     lambda_list=None, fc_sizes=self.fc_sizes, input_shape=self.input_shape)
        
        # Load weights and biases
        param_idx = 5
        for i, layer in enumerate(self.layers):
            if layer.optimizable:
                layer.W = params[param_idx]['W']
                layer.b = params[param_idx]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = params[param_idx]['weight_decay']
                layer.weight_decay_lambda = params[param_idx]['lambda']
                param_idx += 1
        
    def save_model(self, save_path):
        """
        Save model parameters to a file
        save_path: path to save the pickle file
        """
        # Store model architecture and parameters
        param_list = [
            self.channels_list,
            self.kernel_sizes,
            self.act_func,
            self.fc_sizes,
            self.input_shape
        ]
        
        # Store weights and biases of optimizable layers
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
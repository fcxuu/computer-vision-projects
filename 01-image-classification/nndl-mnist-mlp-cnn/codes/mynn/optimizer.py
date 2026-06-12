from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu  # Momentum coefficient
        self.velocities = {}
        
        # Initialize velocity for each layer and parameter
        for i, layer in enumerate(self.model.layers):
            if layer.optimizable:
                self.velocities[i] = {}
                for key in layer.params.keys():
                    self.velocities[i][key] = np.zeros_like(layer.params[key])
    
    def step(self):
        for i, layer in enumerate(self.model.layers):
            if layer.optimizable:
                for key in layer.params.keys():
                    # Update velocity using momentum
                    if layer.weight_decay:
                        weight_decay_grad = layer.weight_decay_lambda * layer.params[key]
                    else:
                        weight_decay_grad = 0
                        
                    # v = mu * v - lr * (grad + weight_decay_grad)
                    self.velocities[i][key] = self.mu * self.velocities[i][key] - self.init_lr * (layer.grads[key] + weight_decay_grad)
                    
                    # Update parameters: params += v
                    layer.params[key] += self.velocities[i][key]
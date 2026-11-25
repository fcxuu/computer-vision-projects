"""Assemble NumPy layers and keep parameter references stable during updates."""
import sys
from pathlib import Path

import numpy as np

from layers import Conv2D, Flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codes"))
from mynn.op import Linear, ReLU, Sigmoid
from mynn.lr_scheduler import ExponentialLR, MultiStepLR, StepLR


class Network:
    def __init__(self, name, activation, seed, mlp_width=600):
        rng = np.random.default_rng(seed)
        activate = {"relu": ReLU, "sigmoid": Sigmoid}[activation]

        def linear(inputs, outputs, relu=False):
            std = np.sqrt((2 if relu else 1) / inputs)

            def initialize(size):
                return (rng.standard_normal(size) * std).astype(np.float32)

            layer = Linear(inputs, outputs, initialize_method=initialize)
            layer.b.fill(0)
            return layer

        if name == "mlp":
            self.layers = [
                linear(784, mlp_width, activation == "relu"), activate(), linear(mlp_width, 10),
            ]
        elif name == "cnn":
            self.layers = [
                Conv2D(1, 16, 5, 2, 2, rng=rng), activate(),
                Conv2D(16, 32, 5, 2, 2, rng=rng), activate(),
                Flatten(), linear(32 * 7 * 7, 100, activation == "relu"), activate(),
                linear(100, 10),
            ]
        else:
            raise ValueError(f"unknown network: {name}")
        self.name = name
        self.activation = activation
        self.mlp_width = mlp_width

    def __call__(self, pixels):
        output = pixels if self.name == "mlp" else pixels.reshape(-1, 1, 28, 28)
        for layer in self.layers:
            output = layer(output)
        return output

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)

    def parameters(self):
        for index, layer in enumerate(self.layers):
            if layer.optimizable:
                for name, value in layer.params.items():
                    yield f"{index}_{name}", value, layer.grads[name]

    def save(self, path):
        partial = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            partial, network=self.name, activation=self.activation, mlp_width=self.mlp_width,
            **{key: value for key, value, _ in self.parameters()},
        )
        partial.replace(path)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            mlp_width = int(archive["mlp_width"].item()) if "mlp_width" in archive else 600
            model = cls(str(archive["network"].item()), str(archive["activation"].item()), 0, mlp_width)
            for key, weight, _ in model.parameters():
                if archive[key].shape != weight.shape or not np.isfinite(archive[key]).all():
                    raise ValueError(f"invalid checkpoint tensor {key}")
                weight[:] = archive[key]
        return model


class GradientDescent:
    """Momentum SGD, with zero momentum implementing ordinary SGD."""
    def __init__(self, model, lr, momentum, l2):
        self.model = model
        self.init_lr = lr
        self.momentum = momentum
        self.l2 = l2
        self.buffers = {key: np.zeros_like(weight) for key, weight, _ in model.parameters()}

    def step(self):
        for key, value, gradient in self.model.parameters():
            if key.endswith("_W"):
                gradient = gradient + self.l2 * value
            velocity = self.buffers[key]
            velocity *= self.momentum
            velocity -= self.init_lr * gradient
            value += velocity


def scheduler(optimizer, mode, epochs):
    if mode == "exponential":
        return ExponentialLR(optimizer, gamma=0.91)
    if mode == "multistep":
        milestones = sorted({max(1, epochs // 2), max(1, 4 * epochs // 5)})
        return MultiStepLR(optimizer, milestones=milestones, gamma=0.4)
    return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.6)

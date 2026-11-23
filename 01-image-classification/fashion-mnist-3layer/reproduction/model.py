"""A three-layer network (input, one hidden layer, output) using NumPy only."""
import os
import tempfile

import numpy as np

PARAMETERS = ("W1", "b1", "W2", "b2")
ACTIVATIONS = ("relu", "sigmoid", "tanh")


def initialize(inputs, hidden, outputs, activation, rng, dtype=np.float32):
    gain = np.sqrt(2.0 if activation == "relu" else 1.0)
    return {
        "W1": (rng.standard_normal((inputs, hidden)) * gain / np.sqrt(inputs)).astype(dtype),
        "b1": np.zeros(hidden, dtype=dtype),
        "W2": (rng.standard_normal((hidden, outputs)) / np.sqrt(hidden)).astype(dtype),
        "b2": np.zeros(outputs, dtype=dtype),
    }


def forward(parameters, x, activation):
    preactivation = x @ parameters["W1"] + parameters["b1"]
    if activation == "relu":
        hidden = np.maximum(0, preactivation)
    elif activation == "sigmoid":
        hidden = np.exp(-np.logaddexp(0, -preactivation))
    elif activation == "tanh":
        hidden = np.tanh(preactivation)
    else:
        raise ValueError(f"Unsupported activation {activation!r}")
    return hidden @ parameters["W2"] + parameters["b2"], hidden


def log_probabilities(logits):
    centered = logits - logits.max(axis=1, keepdims=True)
    return centered - np.log(np.exp(centered).sum(axis=1, keepdims=True))


def l2_penalty(parameters, strength):
    return float(0.5 * strength * sum(np.square(parameters[k]).sum() for k in ("W1", "W2")))


def objective_and_gradients(parameters, x, targets, activation, l2):
    logits, hidden = forward(parameters, x, activation)
    log_probs = log_probabilities(logits)
    rows = np.arange(len(targets))
    cross_entropy = float(-log_probs[rows, targets].mean())
    output_grad = np.exp(log_probs)
    output_grad[rows, targets] -= 1.0
    output_grad /= len(targets)
    if activation == "relu":
        derivative = hidden > 0
    elif activation == "sigmoid":
        derivative = hidden * (1 - hidden)
    else:
        derivative = 1 - hidden**2
    hidden_grad = (output_grad @ parameters["W2"].T) * derivative
    gradients = {
        "W1": x.T @ hidden_grad + l2 * parameters["W1"],
        "b1": hidden_grad.sum(axis=0),
        "W2": hidden.T @ output_grad + l2 * parameters["W2"],
        "b2": output_grad.sum(axis=0),
    }
    return cross_entropy + l2_penalty(parameters, l2), gradients


def save_checkpoint(path, parameters, activation):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        try:
            np.savez_compressed(stream, activation=activation, **parameters)
            stream.flush()
            os.replace(stream.name, path)
        finally:
            if os.path.exists(stream.name):
                os.unlink(stream.name)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as checkpoint:
        activation = str(checkpoint["activation"].item())
        weights = {key: checkpoint[key] for key in PARAMETERS}
    hidden = weights["b1"].size
    shapes = {"W1": (784, hidden), "b1": (hidden,), "W2": (hidden, 10), "b2": (10,)}
    if (
        activation not in ACTIVATIONS or hidden == 0
        or any(weights[k].shape != shape for k, shape in shapes.items())
        or any(not np.isfinite(weight).all() for weight in weights.values())
    ):
        raise ValueError(f"Invalid Fashion-MNIST checkpoint: {path}")
    return weights, activation

"""Numerically verify backpropagation and persistence for the NumPy model."""
import tempfile
from pathlib import Path

import numpy as np

from model import (
    ACTIVATIONS, forward, initialize, load_checkpoint, log_probabilities,
    objective_and_gradients, save_checkpoint,
)


def check_gradients():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(7, 5))
    y = rng.integers(0, 3, size=7)
    epsilon = 1e-5
    for activation in ACTIVATIONS:
        model = initialize(5, 4, 3, activation, rng, dtype=np.float64)
        for l2 in (0.0, 0.03):
            _, gradients = objective_and_gradients(model, x, y, activation, l2)
            max_error = 0.0
            for key, weight in model.items():
                for index in np.ndindex(weight.shape):
                    original = weight[index]
                    weight[index] = original + epsilon
                    upper, _ = objective_and_gradients(model, x, y, activation, l2)
                    weight[index] = original - epsilon
                    lower, _ = objective_and_gradients(model, x, y, activation, l2)
                    weight[index] = original
                    expected = (upper - lower) / (2 * epsilon)
                    max_error = max(max_error, abs(expected - gradients[key][index]))
            if max_error >= 1e-7:
                raise AssertionError(f"{activation}, l2={l2}: error {max_error}")
            print(f"{activation:7s}, l2={l2:.2f}: max gradient error {max_error:.3e}")


def check_checkpoint():
    rng = np.random.default_rng(5)
    model = initialize(784, 3, 10, "sigmoid", rng)
    sample = rng.random((5, 784), dtype=np.float32)
    before, _ = forward(model, sample, "sigmoid")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.npz"
        save_checkpoint(path, model, "sigmoid")
        model["W1"] *= 0
        restored, activation = load_checkpoint(path)
    after, _ = forward(restored, sample, activation)
    np.testing.assert_array_equal(before, after)
    print("NPZ round trip preserved logits and activation")


if __name__ == "__main__":
    check_gradients()
    extreme = np.array([[10_000.0, -10_000.0], [-10_000.0, 10_000.0]])
    log_probs = log_probabilities(extreme)
    assert np.isfinite(log_probs).all()
    np.testing.assert_array_equal(np.exp(log_probs).sum(axis=1), np.ones(2))
    print("Extreme logits: finite cross-entropy and normalized probabilities")
    check_checkpoint()

"""NumPy layers for the NCHW MNIST CNN."""

from operator import index

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Conv2D:
    """Dense, zero-padded 2D cross-correlation with one scalar stride."""

    optimizable = True

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, *, rng, dtype=np.float32):
        self.in_channels = index(in_channels)
        self.out_channels = index(out_channels)
        sizes = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        kh, kw = map(index, sizes)
        self.kernel_size = kh, kw
        self.stride = index(stride)
        self.padding = index(padding)
        if min(self.in_channels, self.out_channels, kh, kw, self.stride) < 1 or self.padding < 0:
            raise ValueError("Channels, kernel dimensions and stride must be positive; padding must be nonnegative")
        dtype = np.dtype(dtype)
        if dtype.kind != "f":
            raise TypeError("Conv2D requires a floating-point dtype")
        scale = np.sqrt(2.0 / (self.in_channels * kh * kw))
        weights = rng.normal(0.0, scale, (self.out_channels, self.in_channels, kh, kw))
        self.params = {"W": weights.astype(dtype), "b": np.zeros(self.out_channels, dtype=dtype)}
        self.grads = {"W": None, "b": None}
        self._cache = None

    @property
    def W(self):
        return self.params["W"]

    @property
    def b(self):
        return self.params["b"]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim != 4 or x.shape[1] != self.in_channels or min(x.shape) < 1:
            raise ValueError(f"Expected a nonempty NCHW input with {self.in_channels} channels")
        batch, channels, height, width = x.shape
        kh, kw = self.kernel_size
        stride, pad = self.stride, self.padding
        if height + 2 * pad < kh or width + 2 * pad < kw:
            raise ValueError("The kernel must fit inside the padded input")
        padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else x
        windows = sliding_window_view(padded, (kh, kw), axis=(2, 3))[:, :, ::stride, ::stride]
        out_h, out_w = windows.shape[2:4]
        # One row per output location, with features ordered C, Kh, Kw.
        columns = np.ascontiguousarray(windows.transpose(0, 2, 3, 1, 4, 5)).reshape(-1, channels * kh * kw)
        output = columns @ self.W.reshape(self.out_channels, -1).T
        output += self.b
        self._cache = (x.shape, out_h, out_w, columns)
        return output.reshape(batch, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

    def backward(self, grad):
        if self._cache is None:
            raise RuntimeError("Conv2D.backward requires a preceding forward pass")
        (batch, channels, height, width), out_h, out_w, columns = self._cache
        grad = np.asarray(grad)
        if grad.shape != (batch, self.out_channels, out_h, out_w):
            raise ValueError("Conv2D output gradient shape does not match the latest forward pass")
        kh, kw = self.kernel_size
        stride, pad = self.stride, self.padding
        rows = np.ascontiguousarray(grad.transpose(0, 2, 3, 1)).reshape(-1, self.out_channels)
        self.grads["W"] = (rows.T @ columns).reshape(self.W.shape)
        self.grads["b"] = rows.sum(axis=0)
        patches = (rows @ self.W.reshape(self.out_channels, -1)).reshape(
            batch, out_h, out_w, channels, kh, kw
        )
        padded_dx = np.zeros((batch, channels, height + 2 * pad, width + 2 * pad), dtype=patches.dtype)
        for row in range(kh):
            for column in range(kw):
                # Locations at a fixed kernel offset are distinct. Offsets can overlap.
                padded_dx[:, :, row:row + out_h * stride:stride, column:column + out_w * stride:stride] += (
                    patches[:, :, :, :, row, column].transpose(0, 3, 1, 2)
                )
        self._cache = None
        return padded_dx[:, :, pad:pad + height, pad:pad + width]

    def clear_grad(self):
        self.grads.update(W=None, b=None)


class Flatten:
    """Flatten all axes after the batch dimension in logical C order."""

    optimizable = False

    def __init__(self):
        self._input_shape = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim < 2 or x.shape[0] == 0:
            raise ValueError("Flatten expects a nonempty batch and at least one feature axis")
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        if self._input_shape is None:
            raise RuntimeError("Flatten.backward requires a preceding forward pass")
        grad = np.asarray(grad)
        expected = (self._input_shape[0], int(np.prod(self._input_shape[1:])))
        if grad.shape != expected:
            raise ValueError(f"Flatten output gradient shape must be {expected}")
        dx = grad.reshape(self._input_shape)
        self._input_shape = None
        return dx

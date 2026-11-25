"""Validate convolution and flattening without external frameworks or data."""

import numpy as np

from layers import Conv2D, Flatten


def naive_conv(layer, x):
    """Independent reference using explicit loops over the output locations."""
    pad, stride = layer.padding, layer.stride
    kh, kw = layer.kernel_size
    batch, _, height, width = x.shape
    out_h = (height + 2 * pad - kh) // stride + 1
    out_w = (width + 2 * pad - kw) // stride + 1
    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.empty((batch, layer.out_channels, out_h, out_w), dtype=x.dtype)
    for sample, channel, row, column in np.ndindex(out.shape):
        top, left = row * stride, column * stride
        patch = padded[sample, :, top:top + kh, left:left + kw]
        out[sample, channel, row, column] = np.sum(patch * layer.W[channel]) + layer.b[channel]
    return out


def central_difference(values, objective, epsilon=1e-5):
    grad = np.empty_like(values)
    for position in np.ndindex(values.shape):
        original = values[position]
        values[position] = original + epsilon
        upper = objective()
        values[position] = original - epsilon
        lower = objective()
        values[position] = original
        grad[position] = (upper - lower) / (2 * epsilon)
    return grad


def check_convolution(rng, kernel, stride, padding):
    conv = Conv2D(2, 3, kernel, stride, padding, rng=rng, dtype=np.float64)
    x = rng.standard_normal((2, 2, 4, 5))[:, :, :, ::-1]
    np.testing.assert_array_equal(conv.b, np.zeros(3))
    conv.b[:] = rng.normal(size=3)
    y = conv(x)
    np.testing.assert_allclose(y, naive_conv(conv, x), atol=1e-12, rtol=1e-12)
    upstream = rng.standard_normal(y.shape)
    dx = conv.backward(upstream)

    def objective():
        return np.sum(conv(x) * upstream)

    errors = []
    for values, analytic in ((conv.W, conv.grads["W"]), (conv.b, conv.grads["b"]), (x, dx)):
        numeric = central_difference(values, objective)
        np.testing.assert_allclose(analytic, numeric, atol=1e-8, rtol=1e-7)
        errors.append(np.max(np.abs(analytic - numeric)))
    singleton = x[:1]
    out = conv.forward(singleton)
    np.testing.assert_allclose(out, naive_conv(conv, singleton), atol=1e-12, rtol=1e-12)
    assert conv.backward(np.ones_like(out)).shape == singleton.shape
    conv.clear_grad()
    assert conv.grads == {"W": None, "b": None}
    print(f"kernel={kernel} stride={stride} padding={padding}: "
          f"max |analytic - numeric| W={errors[0]:.2e}, b={errors[1]:.2e}, X={errors[2]:.2e}")


def check_cnn_shapes(rng):
    first = Conv2D(1, 16, 5, stride=2, padding=2, rng=rng)
    second = Conv2D(16, 32, 5, stride=2, padding=2, rng=rng)
    flatten = Flatten()
    for batch in (8, 3):
        x = rng.standard_normal((batch, 1, 28, 28), dtype=np.float32)
        hidden = first(x)
        assert hidden.shape == (batch, 16, 14, 14)
        activation = second(hidden)
        assert activation.shape == (batch, 32, 7, 7)
        out = flatten(activation)
        assert out.shape == (batch, 1568) and out.dtype == np.float32
        np.testing.assert_array_equal(out, activation.reshape(batch, 1568))
        grad = rng.standard_normal(out.shape, dtype=np.float32)
        restored = flatten.backward(grad)
        np.testing.assert_array_equal(restored, grad.reshape(batch, 32, 7, 7))
        dx = first.backward(second.backward(restored))
        assert dx.shape == x.shape and dx.dtype == np.float32 and np.isfinite(dx).all()
    print("Float32 CNN + Flatten passed for batch sizes 8 and 3")


def main():
    rng = np.random.default_rng(2026)
    for kernel, stride, padding in [(2, 1, 0), ((2, 3), 1, 1), ((3, 2), 2, 0), (3, 2, 2)]:
        check_convolution(rng, kernel, stride, padding)
    check_cnn_shapes(rng)


if __name__ == "__main__":
    main()

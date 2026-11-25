"""Measure a fixed-batch loss surface around a Conv2d/Linear checkpoint."""

import math
from operator import index

import torch
from torch import nn
from torch.nn import functional as F


def _direction(weight, generator):
    """Give each random filter the L2 norm of its corresponding weight filter."""
    filters = weight.detach().cpu().double().flatten(start_dim=1)
    random = torch.randn(filters.shape, dtype=torch.float64, generator=generator)
    scales = filters.norm(dim=1, keepdim=True) / random.norm(dim=1, keepdim=True)
    return (random * scales).reshape_as(weight).to(dtype=weight.dtype).to(device=weight.device)


@torch.no_grad()
def surface(model, batches, *, seed=2020, points=9, radius=0.5):
    """Return sample-mean cross-entropy on two filter-normalized directions.

    Inputs are already normalized and on the model device. The iterable is
    consumed once, so every grid point sees identical batches. Rows correspond
    to axis_y and columns to axis_x. An odd grid includes the exact checkpoint.
    """
    points = index(points)
    radius = float(radius)
    if points < 3 or points % 2 == 0 or not math.isfinite(radius) or radius <= 0:
        raise ValueError("Use an odd grid with at least three points and a finite positive radius")
    batches = tuple(batches)
    sample_count = sum(labels.numel() for _, labels in batches)
    if not sample_count:
        raise ValueError("At least one sample is required")
    modules = tuple(model.modules())
    weights = tuple(dict.fromkeys(
        layer.weight for layer in modules if isinstance(layer, (nn.Conv2d, nn.Linear))
    ))
    if not weights:
        raise ValueError("The model must have a Conv2d or Linear weight")
    modes = tuple(layer.training for layer in modules)
    original = tuple(weight.detach().clone() for weight in weights)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = tuple((_direction(base, generator), _direction(base, generator)) for base in original)
    middle = points // 2
    axis = [radius * ((step - middle) / middle) for step in range(points)]
    losses = []
    try:
        model.eval()
        for y in axis:
            row = []
            for x in axis:
                for weight, base, (dx, dy) in zip(weights, original, directions):
                    weight.copy_(base)
                    if x:
                        weight.add_(dx, alpha=x)
                    if y:
                        weight.add_(dy, alpha=y)
                total = sum(
                    F.cross_entropy(model(images), labels, reduction="sum").item()
                    for images, labels in batches
                )
                mean = total / sample_count
                if not math.isfinite(mean):
                    raise FloatingPointError(f"Non-finite loss at x={x:g}, y={y:g}")
                row.append(mean)
            losses.append(row)
    finally:
        for weight, base in zip(weights, original):
            weight.copy_(base)
        for layer, training in zip(modules, modes):
            layer.training = training
    return {
        "seed": int(seed),
        "points": points,
        "radius": radius,
        "axis_x": axis,
        "axis_y": axis.copy(),
        "losses": losses,
        "origin": losses[middle][middle],
        "minimum": min(map(min, losses)),
        "maximum": max(map(max, losses)),
        "samples": sample_count,
    }


def _loss_gradient(model, weights, batches, samples):
    """Accumulate the dataset gradient one batch at a time without touching .grad."""
    accumulators = [torch.zeros_like(weight) for weight in weights]
    totals = []
    with torch.enable_grad():
        for images, labels in batches:
            loss = F.cross_entropy(model(images), labels, reduction="sum")
            gradients = torch.autograd.grad(loss / samples, weights, allow_unused=True)
            totals.append(loss.detach().item())
            for accumulator, gradient in zip(accumulators, gradients):
                if gradient is not None:
                    accumulator.add_(gradient)
    mean = math.fsum(totals) / samples
    flat = torch.cat([gradient.flatten().cpu() for gradient in accumulators]).double()
    if not math.isfinite(mean) or not torch.isfinite(flat).all():
        raise FloatingPointError("Non-finite loss or gradient in the directional probe")
    return mean, accumulators, flat


def gradient_probe(model, batches, *, relative_radii=(0.0, 0.001, 0.003, 0.01, 0.03)):
    """Measure first-order predictions and gradient change along the center descent ray.

    Relative distances are measured against the joint L2 norm of the Conv2d
    and Linear weights. The gradient-change ratio is null at zero distance.
    The maximum gradient difference is over the sampled points on this ray.
    """
    radii = tuple(map(float, relative_radii))
    if not radii or any(not math.isfinite(radius) or radius < 0 for radius in radii):
        raise ValueError("Specify finite, nonnegative relative radii")
    batches = tuple(batches)
    samples = sum(labels.numel() for _, labels in batches)
    if not samples:
        raise ValueError("At least one sample is required")
    modules = tuple(model.modules())
    weights = tuple(dict.fromkeys(
        layer.weight for layer in modules if isinstance(layer, (nn.Conv2d, nn.Linear))
    ))
    if not weights:
        raise ValueError("The model must have a Conv2d or Linear weight")
    modes = tuple(layer.training for layer in modules)
    requires_grad = tuple(weight.requires_grad for weight in weights)
    original = tuple(weight.detach().clone() for weight in weights)
    buffers = tuple((buffer, buffer.detach().clone()) for buffer in model.buffers())
    weight_norm = math.hypot(*(base.cpu().double().norm().item() for base in original))
    records = []
    try:
        model.eval()
        for weight in weights:
            weight.requires_grad_(True)
        center_loss, center_gradients, center = _loss_gradient(model, weights, batches, samples)
        center_norm = center.norm().item()
        if not math.isfinite(weight_norm) or not math.isfinite(center_norm):
            raise FloatingPointError("Non-finite weight or gradient norm in the directional probe")
        if not center_norm:
            raise ValueError("The center gradient is zero; a normalized descent direction is undefined")

        for radius in radii:
            distance = radius * weight_norm
            if not math.isfinite(distance):
                raise FloatingPointError("Non-finite probe distance")
            if distance:
                with torch.no_grad():
                    for weight, base, gradient in zip(weights, original, center_gradients):
                        weight.copy_(base).add_(gradient, alpha=-distance / center_norm)
                actual, _, gradient = _loss_gradient(model, weights, batches, samples)
                gradient_norm = gradient.norm().item()
                difference = (gradient - center).norm().item()
                cosine = float(gradient.dot(center).item() / (gradient_norm * center_norm)) if gradient_norm else None
                ratio = difference / distance
            else:
                actual = center_loss
                difference, cosine, ratio = 0.0, 1.0, None
            predicted = center_loss - distance * center_norm
            if not all(math.isfinite(value) for value in (actual, predicted, difference, cosine, ratio) if value is not None):
                raise FloatingPointError("Non-finite directional probe measurement")
            records.append({
                "relative_radius": radius,
                "distance": distance,
                "actual_loss": actual,
                "predicted_loss": predicted,
                "gradient_cosine": cosine,
                "gradient_difference": difference,
                "gradient_difference_per_distance": ratio,
            })
    finally:
        with torch.no_grad():
            for weight, base, required in zip(weights, original, requires_grad):
                weight.copy_(base)
                weight.requires_grad_(required)
            for buffer, base in buffers:
                buffer.copy_(base)
        for layer, training in zip(modules, modes):
            layer.training = training
    return {
        "samples": samples,
        "weight_norm": weight_norm,
        "center_loss": center_loss,
        "center_gradient_norm": center_norm,
        "maximum_gradient_difference": max(record["gradient_difference"] for record in records),
        "points": records,
    }

"""Fit coarse and fine NeRFs on registered real photographs."""
import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from field import VENDOR, VENDOR_COMMIT, Field, Scene, device_for


def prepare_rays(scene, split):
    for frame in scene.frames(split):
        origins, directions = scene.rays(frame)
        rgb = torch.from_numpy(scene.rgb(frame)).to(scene.device)
        yield (origins, directions, rgb)


def train_pool(scene):
    groups = tuple(prepare_rays(scene, "train"))
    return tuple(torch.cat([frame[key].reshape(-1, 3) for frame in groups]) for key in range(3))


def focus_indices(origins, directions, radius):
    """Find forward rays passing through a sphere around the calibrated object center."""
    depth = -(origins * directions).sum(dim=1)
    closest = origins + depth[:, None] * directions
    within = (depth > 0) & (closest.square().sum(dim=1) <= radius ** 2)
    return torch.nonzero(within, as_tuple=False).flatten().cpu()


def psnr(mse):
    return -10.0 * math.log10(max(mse, 1e-12))


@torch.no_grad()
def validate(model, views, chunk):
    model.eval()
    mse = []
    for origins, directions, image in views:
        prediction, _ = model.render(origins, directions, chunk)
        mse.append(float(F.mse_loss(prediction, image)))
    model.train()
    return float(np.mean(mse)), float(np.mean([psnr(value) for value in mse]))


def save_checkpoint(path, model, scene, config, step):
    temporary = path.with_suffix(".tmp")
    torch.save({"weights": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "scene_digest": scene.digest, "framework_commit": VENDOR_COMMIT,
                "config": config, "step": step}, temporary)
    temporary.replace(path)


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = device_for(args.device)
    scene = Scene(args.scene, device)
    keys = ("steps", "rays", "n_coarse", "n_fine", "layers", "width", "frequencies_xyz",
            "frequencies_dir", "lr", "seed", "validate_every", "chunk", "focus_share", "focus_radius")
    config = {key: getattr(args, key) for key in keys}
    config["device"] = str(device)
    rays = train_pool(scene)
    validation = tuple(prepare_rays(scene, "val"))
    model = Field(args.vendor, config, scene).train()
    if args.initialize_from:
        source = torch.load(args.initialize_from, map_location="cpu", weights_only=True)
        if source["scene_digest"] != scene.digest or source["framework_commit"] != VENDOR_COMMIT:
            raise ValueError("Initialization checkpoint and scene disagree")
        model.load_state_dict(source["weights"], strict=True)
        config["initialization_step"] = source["step"]
    focus_count = round(args.focus_share * args.rays)
    focus_pool = focus_indices(*rays[:2], args.focus_radius) if focus_count else None
    if focus_pool is not None and not len(focus_pool):
        raise ValueError("Focus sphere intersects no training rays")
    config["focus_pool_fraction"] = len(focus_pool) / len(rays[0]) if focus_pool is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=args.lr * 0.1)
    random = torch.Generator().manual_seed(args.seed + 1)
    args.out.mkdir(parents=True, exist_ok=False)
    directory = args.out / "checkpoints"
    directory.mkdir()
    best_psnr, best_step = -float("inf"), 0
    fields = ("step", "train_mse", "train_psnr", "val_mse", "val_psnr", "lr")
    total = torch.zeros((), device=device)
    count = 0
    started = time.perf_counter()
    with (
        (args.out / "history.csv").open("w", newline="") as file,
        SummaryWriter(str(args.out / "tensorboard")) as writer,
    ):
        history = csv.DictWriter(file, fields, lineterminator="\n")
        history.writeheader()
        for step in range(1, args.steps + 1):
            indices = torch.randint(len(rays[0]), (args.rays - focus_count,), generator=random)
            if focus_pool is not None:
                selected = torch.randint(len(focus_pool), (focus_count,), generator=random)
                indices = torch.cat((indices, focus_pool[selected]))
            indices = indices.to(device)
            origins, directions, rgb = (tensor.index_select(0, indices) for tensor in rays)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(origins, directions)
            fine_loss = F.mse_loss(prediction["rgb_fine"], rgb)
            objective = fine_loss + F.mse_loss(prediction["rgb_coarse"], rgb)
            objective.backward()
            optimizer.step()
            total += fine_loss.detach()
            count += 1
            validate_now = step % args.validate_every == 0 or step == args.steps
            if step % 50 == 0 or validate_now:
                mse = float(total) / count
                if not math.isfinite(mse):
                    raise FloatingPointError(f"Non-finite prediction error at step {step}")
                train_psnr = psnr(mse)
                record = {"step": step, "train_mse": mse, "train_psnr": train_psnr,
                          "lr": optimizer.param_groups[0]["lr"]}
                writer.add_scalar("train/mse", mse, step)
                writer.add_scalar("train/psnr", train_psnr, step)
                if validate_now:
                    val_mse, val_psnr = validate(model, validation, args.chunk)
                    if not all(math.isfinite(value) for value in (val_mse, val_psnr)):
                        raise FloatingPointError(f"Non-finite validation result at step {step}")
                    record.update(val_mse=val_mse, val_psnr=val_psnr)
                    writer.add_scalar("validation/mse", val_mse, step)
                    writer.add_scalar("validation/psnr", val_psnr, step)
                    checkpoint = directory / f"step-{step:06d}.pt"
                    save_checkpoint(checkpoint, model, scene, config, step)
                    if val_psnr > best_psnr:
                        best_psnr, best_step = val_psnr, step
                        temporary = args.out / "best.tmp"
                        shutil.copyfile(checkpoint, temporary)
                        temporary.replace(args.out / "best.pt")
                    print(f"{step}/{args.steps} train {train_psnr:.2f} dB val {val_psnr:.2f} dB "
                          f"({time.perf_counter() - started:.1f}s)", flush=True)
                elif step % 250 == 0:
                    print(f"{step}/{args.steps} train {train_psnr:.2f} dB "
                          f"({time.perf_counter() - started:.1f}s)", flush=True)
                history.writerow(record)
                file.flush()
                writer.flush()
                total.zero_()
                count = 0
            schedule.step()
    result = {"config": config, "best_step": best_step, "best_val_psnr": best_psnr,
              "seconds": time.perf_counter() - started, "scene_digest": scene.digest,
              "framework_commit": VENDOR_COMMIT,
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "training_views": len(scene.frames("train")), "validation_views": len(validation),
              "objective": "sum of coarse and fine RGB reconstruction MSE"}
    (args.out / "train.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Selected checkpoint {best_step} by validation PSNR ({best_psnr:.2f} dB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, default=VENDOR)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    for name, default in (("steps", 7000), ("rays", 384), ("n-coarse", 48), ("n-fine", 32),
                          ("layers", 6), ("width", 96), ("frequencies-xyz", 8), ("frequencies-dir", 3),
                          ("validate-every", 1000), ("chunk", 1024), ("seed", 2020), ("threads", 4)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--focus-share", type=float, default=0.0)
    parser.add_argument("--focus-radius", type=float, default=1.1)
    args = parser.parse_args()
    if min(args.steps, args.rays, args.n_fine, args.frequencies_xyz, args.frequencies_dir, args.validate_every,
           args.chunk, args.threads) < 1 or args.n_coarse < 3 or args.layers < 3 or args.width < 4:
        parser.error("Counts must be positive, with at least three coarse samples and three field layers")
    if not (0 < args.lr < float("inf")) or args.seed < 0:
        parser.error("Use a finite positive learning rate and nonnegative seed")
    if not 0 <= args.focus_share <= 1 or not 0 < args.focus_radius < float("inf"):
        parser.error("Use a focus share in [0, 1] and a finite positive radius")
    train(args)


if __name__ == "__main__":
    main()

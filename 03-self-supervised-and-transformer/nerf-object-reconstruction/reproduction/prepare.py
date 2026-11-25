"""Estimate Buddha Dataset camera parameters from RGB photos using CPU COLMAP."""

import argparse
from collections import Counter
from hashlib import md5
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
from PIL import Image


SOURCE = "https://doi.org/10.5281/zenodo.1203283"
ARCHIVE_MD5 = "65d6ff0eee157cf1174801a36bd60566"
IMAGE_NAME = re.compile(r"[0-9]{5}\._c\.png")


def extract_photos(archive, destination, limit):
    with ZipFile(archive) as source:
        members = {}
        for entry in source.infolist():
            path = PurePosixPath(entry.filename)
            if (
                entry.is_dir()
                or len(path.parts) < 2
                or path.parts[-2] != "buddha"
                or not IMAGE_NAME.fullmatch(path.name)
            ):
                continue
            if path.name in members:
                raise ValueError(f"Duplicate dataset photo: {path.name}")
            members[path.name] = entry
        expected = {f"{number:05d}._c.png" for number in range(1, 68)}
        if members.keys() != expected:
            raise ValueError("Archive must contain all 67 expected Buddha photos")
        destination.mkdir()
        for name, entry in sorted(members.items()):
            with Image.open(BytesIO(source.read(entry))) as image:
                if image.format != "PNG":
                    raise ValueError(f"Expected PNG photo: {name}")
                photo = image.convert("RGB")
                photo.thumbnail((limit, limit), Image.Resampling.LANCZOS)
                photo.save(destination / name)
    return sorted(members)


def fit_scene(poses, points):
    """Find a robust common optical-axis intersection, then check sparse support."""
    origins = poses[:, :3, 3]
    directions = poses[:, :3, 2]
    projection = np.identity(3) - directions[:, :, None] * directions[:, None, :]
    projected_origins = np.einsum("nij,nj->ni", projection, origins)
    weights = np.ones(len(poses))
    for _ in range(4):
        center, _, rank, _ = np.linalg.lstsq(
            (projection * weights[:, None, None]).reshape(-1, 3),
            (projected_origins * weights[:, None]).ravel(),
            rcond=None,
        )
        if rank < 3:
            raise ValueError("Camera axes do not constrain a 3D scene center")
        axis_distance = np.linalg.norm(
            np.einsum("nij,nj->ni", projection, center - origins), axis=1
        )
        cutoff = max(2 * np.median(axis_distance), np.finfo(float).eps)
        weights = np.sqrt(cutoff / np.maximum(axis_distance, cutoff))
    radius = float(np.median(np.linalg.norm(origins - center, axis=1)))
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("The camera ring has no finite positive radius")
    scale = 4 / radius
    depths = np.einsum("ni,ni->n", center - origins, directions) * scale
    if np.any(depths <= 0):
        raise ValueError("Estimated object center lies behind a registered camera")
    point_distances = np.linalg.norm(points - center, axis=1) * scale
    # A cropped central region limits the influence of the table and distant background.
    local = point_distances[point_distances <= 3]
    if len(local) < 10:
        raise ValueError("Too few central sparse points to check the object extent")
    extent = max(1.0, float(np.quantile(local, 0.98)) * 1.15)
    near = max(0.05, float(depths.min()) - extent)
    far = float(depths.max()) + extent
    return center, scale, near, far, {
        "median_axis_distance": float(np.median(axis_distance) * scale),
        "median_camera_radius": 4.0,
        "central_sparse_points": int(len(local)),
        "central_sparse_radius_p98": float(np.quantile(local, 0.98)),
        "depth_margin": extent,
    }


def pose_matrix(image):
    pose = np.identity(4)
    pose[:3, :] = image.cam_from_world().inverse().matrix()
    return pose


def split_for(index):
    return {0: "test", 4: "val"}.get(index % 8, "train")


def export_scene(out, reconstruction, source_names, render_size, metadata):
    registered = sorted(
        (reconstruction.images[image_id] for image_id in reconstruction.reg_image_ids()),
        key=lambda image: image.name,
    )
    if len(registered) < 12:
        raise ValueError(f"Only {len(registered)} photos were registered")
    poses = np.stack([pose_matrix(image) for image in registered])
    points = np.asarray([point.xyz for point in reconstruction.points3D.values()])
    if points.shape != (len(points), 3) or not len(points):
        raise ValueError("COLMAP produced no sparse 3D points")
    center, scale, near, far, diagnostics = fit_scene(poses, points)
    poses[:, :3, 3] = (poses[:, :3, 3] - center) * scale

    first = reconstruction.cameras[registered[0].camera_id]
    ratio = render_size / max(first.width, first.height)
    width = max(1, round(first.width * ratio))
    height = max(1, round(first.height * ratio))
    target = out / "images"
    target.mkdir()
    frames = []
    for index, (image, pose) in enumerate(zip(registered, poses, strict=True)):
        name = image.name
        camera = reconstruction.cameras[image.camera_id]
        if IMAGE_NAME.fullmatch(name) is None or name not in source_names:
            raise ValueError(f"Unexpected image in the COLMAP model: {name}")
        if camera.model.name not in {"PINHOLE", "SIMPLE_PINHOLE"}:
            raise ValueError(f"Unexpected camera model after undistortion: {camera.model}")
        with Image.open(out / "undistorted" / "images" / name) as photo:
            if photo.size != (camera.width, camera.height):
                raise ValueError(f"Undistorted image dimensions differ from intrinsics: {name}")
            photo.convert("RGB").resize((width, height), Image.Resampling.LANCZOS).save(target / name)
        sx = width / camera.width
        sy = height / camera.height
        frames.append({
            "name": name,
            "file_path": f"images/{name}",
            "camera_to_world": pose.tolist(),
            "intrinsics": [
                camera.focal_length_x * sx, camera.focal_length_y * sy,
                camera.principal_point_x * sx, camera.principal_point_y * sy,
            ],
            "split": split_for(index),
        })

    scene = {
        "frames": frames,
        "width": width,
        "height": height,
        "near": near,
        "far": far,
        "registration": {
            "source_frames": len(source_names),
            "registered_frames": len(registered),
            "registered_fraction": len(registered) / len(source_names),
            "split_counts": dict(Counter(frame["split"] for frame in frames)),
            "points3D": len(points),
            "mean_reprojection_error": metadata.pop("mean_reprojection_error"),
        },
        "normalization": {
            "method": "Robust optical-axis intersection; central sparse points check extent",
            "source_center": center.tolist(),
            "scale": scale,
            "transform": "normalized_position = scale * (colmap_position - source_center)",
            **diagnostics,
        },
        "camera_convention": "OpenCV camera-to-world; +X right, +Y down, +Z forward",
        "split_rule": "Registered names sorted lexically; index mod 8: 0 test, 4 val, other train",
        "pose_estimation": "All registered photos participate in geometric pose estimation. "
        "The split is for NeRF photometric training and evaluation.",
        **metadata,
    }
    (out / "scene.json").write_text(
        json.dumps(scene, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(scene["registration"], indent=2))


def match_database(database):
    import pycolmap

    options = pycolmap.FeatureMatchingOptions()
    options.num_threads = 1
    options.use_gpu = False
    pycolmap.match_exhaustive(database, matching_options=options, device=pycolmap.Device.cpu)


def prepare(args):
    import pycolmap

    if pycolmap.__version__ != "4.2.1":
        raise ValueError(f"Expected PyCOLMAP 4.2.1, found {pycolmap.__version__}")
    with args.archive.open("rb") as handle:
        checksum = md5(usedforsecurity=False)
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    if checksum.hexdigest() != ARCHIVE_MD5:
        raise ValueError(f"Unexpected Buddha archive MD5: {checksum.hexdigest()}")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"Output directory must be empty: {out}")
    photos = out / "feature_images"
    source_names = extract_photos(args.archive, photos, args.feature_size)
    database = out / "database.db"
    models = out / "models"
    models.mkdir()
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = args.threads
    extraction.max_image_size = args.feature_size
    extraction.use_gpu = False
    mapping = pycolmap.IncrementalPipelineOptions()
    mapping.num_threads = args.threads
    mapping.random_seed = 0
    mapping.ba_use_gpu = False
    mapping.use_prior_position = False
    pycolmap.extract_features(
        database, photos, camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=pycolmap.ImageReaderOptions(camera_model="SIMPLE_RADIAL"),
        extraction_options=extraction, device=pycolmap.Device.cpu,
    )
    # A separate single-threaded matcher closes the SQLite handles between stages.
    subprocess.run(
        [sys.executable, "-B", "-c", "import prepare, sys; prepare.match_database(sys.argv[1])", str(database)],
        cwd=Path(__file__).resolve().parent, check=True,
    )
    reconstructions = pycolmap.incremental_mapping(database, photos, models, options=mapping)
    if not reconstructions:
        raise ValueError("COLMAP could not initialize a sparse model")
    best = max(reconstructions.values(), key=lambda model: model.num_reg_images())
    sparse = out / "sparse"
    sparse.mkdir()
    best.write(sparse)
    undistorted = out / "undistorted"
    pycolmap.undistort_images(undistorted, sparse, photos, num_threads=args.threads)
    reconstruction = pycolmap.Reconstruction(undistorted / "sparse")
    export_scene(out, reconstruction, source_names, args.render_size, {
        "source": {"doi": SOURCE, "archive_md5": ARCHIVE_MD5, "license": "CC BY 4.0"},
        "colmap": {
            "pycolmap_version": pycolmap.__version__, "device": "cpu",
            "feature_size": args.feature_size, "render_size": args.render_size,
            "threads": args.threads, "matching_threads": 1,
            "features": "SIFT", "matching": "exhaustive",
            "camera_mode": "SINGLE", "camera_model": "SIMPLE_RADIAL",
            "mapper": "incremental", "mapper_random_seed": 0,
            "sparse_model": "sparse", "undistorted_model": "undistorted/sparse",
        },
        "mean_reprojection_error": float(best.compute_mean_reprojection_error()),
    })


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-size", type=positive, default=960)
    parser.add_argument("--render-size", type=positive, default=192)
    parser.add_argument("--threads", type=positive, default=4)
    prepare(parser.parse_args())

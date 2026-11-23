"""Read the official Fashion-MNIST IDX files and isolate the test split."""
import gzip
import hashlib
import shutil
import struct
import tempfile
import urllib.request

import numpy as np

BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
DIGESTS = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
LABELS = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest()


def checked_file(root, name, base_url=BASE_URL, digests=DIGESTS):
    path = root / name
    if path.exists() and digest(path) == digests[name]:
        return path
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
        partial = root / stream.name
        try:
            with urllib.request.urlopen(f"{base_url}/{name}", timeout=60) as source:
                shutil.copyfileobj(source, stream)
            stream.flush()
            if digest(partial) != digests[name]:
                raise ValueError(f"Unexpected IDX checksum for {name}")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    return path


def read_idx(root, split, count, base_url, digests, categories):
    images = checked_file(root, f"{split}-images-idx3-ubyte.gz", base_url, digests)
    labels = checked_file(root, f"{split}-labels-idx1-ubyte.gz", base_url, digests)
    with gzip.open(images, "rb") as stream:
        if struct.unpack(">4I", stream.read(16)) != (2051, count, 28, 28):
            raise ValueError(f"Invalid IDX image header in {images}")
        pixels = np.frombuffer(stream.read(), dtype=np.uint8).reshape(count, 784)
    with gzip.open(labels, "rb") as stream:
        if struct.unpack(">2I", stream.read(8)) != (2049, count):
            raise ValueError(f"Invalid IDX label header in {labels}")
        targets = np.frombuffer(stream.read(), dtype=np.uint8).astype(np.int64)
    if len(targets) != count or np.any(targets >= categories):
        raise ValueError(f"Invalid labels in {labels}")
    return pixels.astype(np.float32) / 255.0, targets


def load(root, split):
    count = {"train": 60_000, "t10k": 10_000}[split]
    return read_idx(root, split, count, BASE_URL, DIGESTS, len(LABELS))


def stratified_split(labels, seed):
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for category in range(len(LABELS)):
        indices = rng.permutation(np.flatnonzero(labels == category))
        validation.append(indices[:600])
        train.append(indices[600:])
    return rng.permutation(np.concatenate(train)), rng.permutation(np.concatenate(validation))

"""Load embeddings and assign duplicate-T group ids."""

from pathlib import Path

import numpy as np

from . import SEED

np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data" / "embeddings.npz"


def load_data(path=None):
    """Load T, I, Y from the embeddings npz as float32 / int arrays."""
    path = Path(path) if path is not None else DEFAULT_PATH
    data = np.load(path)
    T = np.asarray(data["T"], dtype=np.float32)
    I = np.asarray(data["I"], dtype=np.float32)
    Y = np.asarray(data["Y"], dtype=np.int64)
    return T, I, Y


def group_ids(T):
    """Assign an integer group id to each row; identical T (to 6 decimals) share an id."""
    T_rounded = np.round(np.asarray(T, dtype=np.float64), decimals=6)
    _, groups = np.unique(T_rounded, axis=0, return_inverse=True)
    return groups.astype(np.int64)

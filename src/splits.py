"""Train/test splits: group-aware and random stratified."""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from . import SEED

np.random.seed(SEED)


def group_split(Y, groups, test_size=0.2, seed=SEED):
    """Stratified group split so no T-group appears in both train and test.

    Uses StratifiedGroupKFold with n_splits = round(1 / test_size) and returns
    the first fold as the test set.
    """
    Y = np.asarray(Y)
    groups = np.asarray(groups)
    n_splits = max(2, int(round(1.0 / test_size)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_dummy = np.zeros(len(Y))
    train_idx, test_idx = next(sgkf.split(X_dummy, Y, groups))
    return train_idx.astype(np.int64), test_idx.astype(np.int64)


def random_split(Y, test_size=0.2, seed=SEED):
    """Stratified random split (ignores groups; for leakage comparison)."""
    Y = np.asarray(Y)
    indices = np.arange(len(Y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=Y,
        random_state=seed,
    )
    return train_idx.astype(np.int64), test_idx.astype(np.int64)

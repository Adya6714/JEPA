"""Learn g: I -> T under several objectives; evaluate on group-split test sets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import group_ids, load_data
from src.metrics import retrieval
from src.splits import group_split

SEEDS = list(range(5))
METHODS = ["procrustes", "ridge", "mlp_cos", "mlp_vicreg", "infonce"]
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
EPOCHS = 200
BATCH_SIZE = 256
LR = 1e-3
TEMP = 0.07
VICREG_VAR = 0.1
VICREG_COV = 0.01

RESULTS_PATH = ROOT / "results" / "02_mapping.json"
FIG_PATH = ROOT / "figs" / "02_mapping.png"

METRIC_KEYS = [
    "paired_cosine",
    "mse",
    "instance_R@1",
    "instance_R@5",
    "class_R@1",
    "class_R@5",
    "effective_rank",
    "probe_acc",
]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, dim: int = 512, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Mapping methods
# ---------------------------------------------------------------------------


def fit_procrustes(I_train, T_train):
    mu_i = I_train.mean(axis=0)
    mu_t = T_train.mean(axis=0)
    A = I_train - mu_i
    B = T_train - mu_t
    R, _ = orthogonal_procrustes(A, B)

    def predict(I):
        return (I - mu_i) @ R + mu_t

    return predict


def fit_ridge(I_train, T_train):
    x_scaler = StandardScaler().fit(I_train)
    y_scaler = StandardScaler().fit(T_train)
    X = x_scaler.transform(I_train)
    Y = y_scaler.transform(T_train)
    model = RidgeCV(alphas=ALPHAS, cv=5)
    model.fit(X, Y)

    def predict(I):
        return y_scaler.inverse_transform(model.predict(x_scaler.transform(I)))

    return predict


def _train_torch(
    model: nn.Module,
    I_train: np.ndarray,
    T_train: np.ndarray,
    loss_fn,
    seed: int,
    epochs: int = EPOCHS,
    standardize_targets: bool = True,
):
    set_seed(seed)
    device = get_device()

    x_scaler = StandardScaler().fit(I_train)
    I_s = x_scaler.transform(I_train).astype(np.float32)
    if standardize_targets:
        y_scaler = StandardScaler().fit(T_train)
        T_s = y_scaler.transform(T_train).astype(np.float32)
    else:
        y_scaler = None
        T_s = T_train.astype(np.float32)

    ds = TensorDataset(torch.from_numpy(I_s), torch.from_numpy(T_s))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    for _ in range(epochs):
        for xb, tb in loader:
            xb = xb.to(device)
            tb = tb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, tb)
            loss.backward()
            opt.step()

    model.eval()

    @torch.no_grad()
    def predict(I):
        Xs = x_scaler.transform(I).astype(np.float32)
        out = []
        for start in range(0, len(Xs), BATCH_SIZE):
            batch = torch.from_numpy(Xs[start : start + BATCH_SIZE]).to(device)
            pred = model(batch).cpu().numpy()
            out.append(pred)
        pred = np.concatenate(out, axis=0)
        if y_scaler is not None:
            pred = y_scaler.inverse_transform(pred)
        return pred

    return predict


def cosine_loss(pred, target):
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    return (1.0 - (pred_n * target_n).sum(dim=-1)).mean()


def vicreg_loss(pred, target, var_w=VICREG_VAR, cov_w=VICREG_COV):
    cos = cosine_loss(pred, target)
    # Variance hinge: std per dim of g(I) >= 1 (targets are standardized)
    std = pred.std(dim=0)
    var = torch.relu(1.0 - std).mean()
    # Off-diagonal covariance penalty
    z = pred - pred.mean(dim=0)
    n = z.shape[0]
    cov = (z.T @ z) / max(n - 1, 1)
    off = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    cov_pen = off / pred.shape[1]
    return cos + var_w * var + cov_w * cov_pen


def infonce_loss(pred, target, temperature=TEMP):
    z_i = F.normalize(pred, dim=-1)
    z_t = F.normalize(target, dim=-1)
    logits = z_i @ z_t.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def fit_mlp_cos(I_train, T_train, seed):
    return _train_torch(MLP(), I_train, T_train, cosine_loss, seed)


def fit_mlp_vicreg(I_train, T_train, seed):
    return _train_torch(MLP(), I_train, T_train, vicreg_loss, seed)


def fit_infonce(I_train, T_train, seed):
    # Linear map W: 512 -> 512; InfoNCE on normalized embeddings (no y inverse scale needed
    # for geometry, but we still standardize I/T on train for stable training).
    model = nn.Linear(512, 512, bias=False)
    return _train_torch(model, I_train, T_train, infonce_loss, seed, standardize_targets=True)


FITTERS = {
    "procrustes": lambda I, T, seed: fit_procrustes(I, T),
    "ridge": lambda I, T, seed: fit_ridge(I, T),
    "mlp_cos": fit_mlp_cos,
    "mlp_vicreg": fit_mlp_vicreg,
    "infonce": fit_infonce,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def paired_cosine(A: np.ndarray, B: np.ndarray) -> float:
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return float((A * B).sum(axis=1).mean())


def effective_rank(X: np.ndarray) -> float:
    """Entropy of normalized squared singular values after centering."""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s.astype(np.float64) ** 2
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def linear_probe(Z_train, Y_train, Z_test, Y_test) -> float:
    scaler = StandardScaler().fit(Z_train)
    clf = LogisticRegression(max_iter=3000)
    clf.fit(scaler.transform(Z_train), Y_train)
    return float(clf.score(scaler.transform(Z_test), Y_test))


def evaluate(predict, I_train, T_train, Y_train, I_test, T_test, Y_test) -> dict:
    Z_train = np.asarray(predict(I_train), dtype=np.float64)
    Z_test = np.asarray(predict(I_test), dtype=np.float64)
    T_test_f = np.asarray(T_test, dtype=np.float64)

    ret = retrieval(Z_test, T_test_f, Y_test, Y_test)
    return {
        "paired_cosine": paired_cosine(Z_test, T_test_f),
        "mse": float(np.mean((Z_test - T_test_f) ** 2)),
        "instance_R@1": ret["instance_R@1"],
        "instance_R@5": ret["instance_R@5"],
        "class_R@1": ret["class_R@1"],
        "class_R@5": ret["class_R@5"],
        "effective_rank": effective_rank(Z_test),
        "probe_acc": linear_probe(Z_train, Y_train, Z_test, Y_test),
    }


def print_table(summary: dict) -> None:
    header = f"{'method':<14s}" + "".join(f"{k:>16s}" for k in METRIC_KEYS)
    print("\n" + header)
    print("-" * len(header))
    for method in METHODS:
        cells = []
        for key in METRIC_KEYS:
            m = summary[method][key]["mean"]
            s = summary[method][key]["std"]
            cells.append(f"{m:.3f}±{s:.3f}")
        print(f"{method:<14s}" + "".join(f"{c:>16s}" for c in cells))


def plot_metrics(summary: dict, path: Path) -> None:
    n_metrics = len(METRIC_KEYS)
    ncols = 4
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows))
    axes = np.asarray(axes).ravel()
    x = np.arange(len(METHODS))
    for i, key in enumerate(METRIC_KEYS):
        ax = axes[i]
        means = [summary[m][key]["mean"] for m in METHODS]
        stds = [summary[m][key]["std"] for m in METHODS]
        ax.bar(x, means, yerr=stds, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, rotation=30, ha="right", fontsize=8)
        ax.set_title(key)
        ax.set_ylabel(key)
    for j in range(n_metrics, len(axes)):
        axes[j].axis("off")
    fig.suptitle("I→T mapping methods (group split, mean±std over seeds)", y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    T, I, Y = load_data()
    groups = group_ids(T)
    device = get_device()
    print(f"Device: {device}")

    # results[method][metric] = list over seeds
    raw = {m: {k: [] for k in METRIC_KEYS} for m in METHODS}
    per_seed = []

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        set_seed(seed)
        train_idx, test_idx = group_split(Y, groups, test_size=0.2, seed=seed)
        I_tr, T_tr, Y_tr = I[train_idx], T[train_idx], Y[train_idx]
        I_te, T_te, Y_te = I[test_idx], T[test_idx], Y[test_idx]

        seed_row = {"seed": seed}
        for method in METHODS:
            print(f"  fitting {method}...")
            predict = FITTERS[method](I_tr, T_tr, seed)
            metrics = evaluate(predict, I_tr, T_tr, Y_tr, I_te, T_te, Y_te)
            seed_row[method] = metrics
            for k, v in metrics.items():
                raw[method][k].append(v)
            print(
                f"    cos={metrics['paired_cosine']:.3f} mse={metrics['mse']:.4f} "
                f"R@1={metrics['instance_R@1']:.3f} probe={metrics['probe_acc']:.3f} "
                f"erank={metrics['effective_rank']:.2f}"
            )
        per_seed.append(seed_row)

    summary = {}
    for method in METHODS:
        summary[method] = {}
        for key in METRIC_KEYS:
            vals = np.asarray(raw[method][key], dtype=np.float64)
            summary[method][key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "per_seed": vals.tolist(),
            }

    payload = {"seeds": SEEDS, "methods": METHODS, "summary": summary, "per_seed": per_seed}
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {RESULTS_PATH}")

    print_table(summary)
    plot_metrics(summary, FIG_PATH)
    print(f"Wrote {FIG_PATH}")


if __name__ == "__main__":
    main()

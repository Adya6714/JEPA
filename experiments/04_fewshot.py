"""Few-shot linear probes on I, T, and I→T mapped features."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import group_ids, load_data
from src.splits import group_split

_spec = importlib.util.spec_from_file_location(
    "mapping02", ROOT / "experiments" / "02_mapping.py"
)
m02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m02)

SEEDS = list(range(5))
N_DRAWS = 10
K_SHOTS = [1, 5, 10]
FEATURE_NAMES = ["I", "T", "g_proc", "g_ridge", "I_g_ridge"]
RESULTS_PATH = ROOT / "results" / "04_fewshot.json"
FIG_PATH = ROOT / "figs" / "04_fewshot.png"


def sample_fewshot_indices(Y_train: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample k labeled examples per class from the train split."""
    chosen = []
    for c in np.unique(Y_train):
        idx = np.where(Y_train == c)[0]
        if len(idx) < k:
            raise RuntimeError(f"Class {c} has only {len(idx)} train rows; need k={k}")
        chosen.append(rng.choice(idx, size=k, replace=False))
    return np.concatenate(chosen)


def fewshot_acc(X_train_fs, Y_train_fs, X_test, Y_test) -> float:
    scaler = StandardScaler().fit(X_train_fs)
    clf = LogisticRegression(max_iter=3000)
    clf.fit(scaler.transform(X_train_fs), Y_train_fs)
    return float(clf.score(scaler.transform(X_test), Y_test))


def build_features(I, T, g_proc, g_ridge, train_t_mean):
    """Center T and g(*) by train T mean; leave I as-is."""
    T_c = T - train_t_mean
    gp = g_proc - train_t_mean
    gr = g_ridge - train_t_mean
    return {
        "I": I,
        "T": T_c,
        "g_proc": gp,
        "g_ridge": gr,
        "I_g_ridge": np.concatenate([I, gr], axis=1),
    }


def main():
    T, I, Y = load_data()
    groups = group_ids(T)
    classes = np.unique(Y)
    n_classes = len(classes)
    print(f"n={len(Y)} classes={n_classes} draws/seed={N_DRAWS}")

    # scores[feature][k] = list over 50 runs
    scores = {f: {k: [] for k in K_SHOTS} for f in FEATURE_NAMES}
    per_run = []

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        m02.set_seed(seed)
        train_idx, test_idx = group_split(Y, groups, test_size=0.2, seed=seed)
        I_tr, T_tr, Y_tr = I[train_idx], T[train_idx], Y[train_idx]
        I_te, T_te, Y_te = I[test_idx], T[test_idx], Y[test_idx]
        train_t_mean = T_tr.mean(axis=0)

        # Fit maps on ALL train pairs
        pred_proc = m02.fit_procrustes(I_tr, T_tr)
        pred_ridge = m02.fit_ridge(I_tr, T_tr)
        g_proc_tr = np.asarray(pred_proc(I_tr), dtype=np.float64)
        g_ridge_tr = np.asarray(pred_ridge(I_tr), dtype=np.float64)
        g_proc_te = np.asarray(pred_proc(I_te), dtype=np.float64)
        g_ridge_te = np.asarray(pred_ridge(I_te), dtype=np.float64)

        feats_tr = build_features(I_tr, T_tr, g_proc_tr, g_ridge_tr, train_t_mean)
        feats_te = build_features(I_te, T_te, g_proc_te, g_ridge_te, train_t_mean)

        for draw in range(N_DRAWS):
            rng = np.random.default_rng(seed * 1000 + draw)
            run_row = {"seed": seed, "draw": draw, "ks": {}}
            for k in K_SHOTS:
                fs_idx = sample_fewshot_indices(Y_tr, k, rng)
                Y_fs = Y_tr[fs_idx]
                run_row["ks"][str(k)] = {}
                for fname in FEATURE_NAMES:
                    acc = fewshot_acc(
                        feats_tr[fname][fs_idx],
                        Y_fs,
                        feats_te[fname],
                        Y_te,
                    )
                    scores[fname][k].append(acc)
                    run_row["ks"][str(k)][fname] = acc
                print(
                    f"  draw={draw} k={k:2d} "
                    + " ".join(
                        f"{f}={run_row['ks'][str(k)][f]:.3f}" for f in FEATURE_NAMES
                    )
                )
            per_run.append(run_row)

    summary = {}
    for fname in FEATURE_NAMES:
        summary[fname] = {}
        for k in K_SHOTS:
            arr = np.asarray(scores[fname][k], dtype=np.float64)
            summary[fname][str(k)] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "per_run": arr.tolist(),
                "n": int(arr.size),
            }

    payload = {
        "seeds": SEEDS,
        "n_draws": N_DRAWS,
        "n_runs": len(SEEDS) * N_DRAWS,
        "k_shots": K_SHOTS,
        "n_classes": int(n_classes),
        "features": FEATURE_NAMES,
        "summary": summary,
        "per_run": per_run,
        "chance": 1.0 / n_classes,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {RESULTS_PATH}")

    # Table
    print(f"\n{'feature':<12s}" + "".join(f"{'k='+str(k):>16s}" for k in K_SHOTS))
    print("-" * (12 + 16 * len(K_SHOTS)))
    for fname in FEATURE_NAMES:
        cells = []
        for k in K_SHOTS:
            s = summary[fname][str(k)]
            cells.append(f"{s['mean']:.3f}±{s['std']:.3f}")
        print(f"{fname:<12s}" + "".join(f"{c:>16s}" for c in cells))

    # Line plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for fname in FEATURE_NAMES:
        means = [summary[fname][str(k)]["mean"] for k in K_SHOTS]
        stds = [summary[fname][str(k)]["std"] for k in K_SHOTS]
        ax.errorbar(K_SHOTS, means, yerr=stds, marker="o", capsize=3, label=fname)
    ax.axhline(1.0 / n_classes, color="gray", ls="--", lw=1, label="chance 1/31")
    ax.set_xlabel("Labeled examples per class (k)")
    ax.set_ylabel("Test accuracy")
    ax.set_xticks(K_SHOTS)
    ax.set_ylim(0, 1.05)
    ax.set_title("Few-shot linear probes (mean±std over 50 runs)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote {FIG_PATH}")


if __name__ == "__main__":
    main()

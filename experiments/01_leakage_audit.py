"""Leakage audit: random vs group split classification accuracy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import group_ids, load_data
from src.splits import group_split, random_split

SEEDS = list(range(5))
SPLIT_TYPES = ["random", "group"]
FEATURE_NAMES = ["T", "I", "TI"]
MODELS = {
    "LogisticRegression": lambda: LogisticRegression(max_iter=3000),
    "KNeighborsClassifier": lambda: KNeighborsClassifier(n_neighbors=5, metric="cosine"),
}

RESULTS_PATH = ROOT / "results" / "01_leakage.json"
FIG_PATH = ROOT / "figs" / "01_leakage.png"


def feature_matrix(name: str, T: np.ndarray, I: np.ndarray) -> np.ndarray:
    if name == "T":
        return T
    if name == "I":
        return I
    if name == "TI":
        return np.concatenate([T, I], axis=1)
    raise ValueError(f"Unknown feature: {name}")


def exact_t_leak_fraction(T: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    """Fraction of test rows whose exact T vector also appears in train."""
    T = np.ascontiguousarray(T)
    train_keys = {row.tobytes() for row in T[train_idx]}
    return float(np.mean([row.tobytes() in train_keys for row in T[test_idx]]))


def run_one(X, Y, train_idx, test_idx, model_factory):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    clf = model_factory()
    clf.fit(X_train, Y[train_idx])
    return float(clf.score(X_test, Y[test_idx]))


def main():
    T, I, Y = load_data()
    groups = group_ids(T)

    # accuracies[feature][model][split_type] = list of scores over seeds
    accuracies = {
        feat: {model: {st: [] for st in SPLIT_TYPES} for model in MODELS}
        for feat in FEATURE_NAMES
    }
    leak_fractions = []

    for seed in SEEDS:
        for split_type in SPLIT_TYPES:
            if split_type == "random":
                train_idx, test_idx = random_split(Y, test_size=0.2, seed=seed)
                leak_fractions.append(exact_t_leak_fraction(T, train_idx, test_idx))
            else:
                train_idx, test_idx = group_split(Y, groups, test_size=0.2, seed=seed)

            for feat in FEATURE_NAMES:
                X = feature_matrix(feat, T, I)
                for model_name, factory in MODELS.items():
                    acc = run_one(X, Y, train_idx, test_idx, factory)
                    accuracies[feat][model_name][split_type].append(acc)
                    print(
                        f"seed={seed} split={split_type:6s} feat={feat:2s} "
                        f"model={model_name:22s} acc={acc:.4f}"
                    )

    # Aggregate summary for table / plot
    summary_rows = []
    for feat in FEATURE_NAMES:
        for model_name in MODELS:
            random_scores = np.asarray(accuracies[feat][model_name]["random"])
            group_scores = np.asarray(accuracies[feat][model_name]["group"])
            random_mean, random_std = float(random_scores.mean()), float(random_scores.std())
            group_mean, group_std = float(group_scores.mean()), float(group_scores.std())
            drop = random_mean - group_mean
            summary_rows.append(
                {
                    "feature": feat,
                    "model": model_name,
                    "random_mean": random_mean,
                    "random_std": random_std,
                    "group_mean": group_mean,
                    "group_std": group_std,
                    "drop": drop,
                }
            )

    payload = {
        "seeds": SEEDS,
        "leak_fraction_random": {
            "per_seed": leak_fractions,
            "mean": float(np.mean(leak_fractions)),
            "std": float(np.std(leak_fractions)),
        },
        "accuracies": accuracies,
        "summary": summary_rows,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {RESULTS_PATH}")

    # Table
    print(
        f"\n{'feature x model':<40s} {'random mean±std':>16s} "
        f"{'group mean±std':>16s} {'drop':>8s}"
    )
    print("-" * 84)
    for row in summary_rows:
        label = f"{row['feature']} x {row['model']}"
        random_s = f"{row['random_mean']:.3f}±{row['random_std']:.3f}"
        group_s = f"{row['group_mean']:.3f}±{row['group_std']:.3f}"
        print(f"{label:<40s} {random_s:>16s} {group_s:>16s} {row['drop']:8.3f}")

    leak = payload["leak_fraction_random"]
    print(
        f"\nRandom-split T leakage fraction: "
        f"{leak['mean']:.3f}±{leak['std']:.3f} (per seed: {leak['per_seed']})"
    )

    # Grouped bar chart
    labels = [f"{r['feature']}\n{r['model'].replace('Classifier', '').replace('Regression', '')}" for r in summary_rows]
    x = np.arange(len(summary_rows))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        [r["random_mean"] for r in summary_rows],
        width,
        yerr=[r["random_std"] for r in summary_rows],
        label="random",
        capsize=3,
    )
    ax.bar(
        x + width / 2,
        [r["group_mean"] for r in summary_rows],
        width,
        yerr=[r["group_std"] for r in summary_rows],
        label="group",
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Leakage audit: random vs group split")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote {FIG_PATH}")


if __name__ == "__main__":
    main()

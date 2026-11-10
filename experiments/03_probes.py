"""Probe suite: within-class residuals, complementarity, CSLS, space agreement."""

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
from src.metrics import _l2_normalize
from src.splits import group_split

_spec = importlib.util.spec_from_file_location(
    "mapping02", ROOT / "experiments" / "02_mapping.py"
)
m02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m02)

SEEDS = list(range(5))
K_CSLS = 10
K_MUTUAL = 10
RESULTS_PATH = ROOT / "results" / "03_probes.json"
FIGS = ROOT / "figs"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def class_means(X: np.ndarray, Y: np.ndarray) -> dict:
    means = {}
    for c in np.unique(Y):
        means[int(c)] = X[Y == c].mean(axis=0)
    return means


def apply_class_means(X: np.ndarray, Y: np.ndarray, means: dict) -> np.ndarray:
    R = np.empty_like(X, dtype=np.float64)
    for i, y in enumerate(Y):
        R[i] = X[i] - means[int(y)]
    return R


def mean_std(vals: list[float]) -> dict:
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "per_seed": arr.tolist()}


def cosine_sims(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    return _l2_normalize(query) @ _l2_normalize(gallery).T


def csls_sims(query: np.ndarray, gallery: np.ndarray, k: int = K_CSLS) -> np.ndarray:
    """CSLS: 2 * cos / (r_q + r_g), r = similarity to k-th NN in the other set."""
    sims = cosine_sims(query, gallery)
    nq, ng = sims.shape
    kq = min(k, ng)
    kg = min(k, nq)
    # k-th largest per row/col
    r_q = np.partition(sims, -kq, axis=1)[:, -kq:].min(axis=1)
    r_g = np.partition(sims.T, -kg, axis=1)[:, -kg:].min(axis=1)
    return 2.0 * sims / (r_q[:, None] + r_g[None, :] + 1e-12)


def retrieval_from_sims(sims: np.ndarray, query_labels, gallery_labels, groups=None) -> dict:
    ranked = np.argsort(-sims, axis=1)
    n_query = sims.shape[0]
    targets = np.arange(n_query)
    query_labels = np.asarray(query_labels)
    gallery_labels = np.asarray(gallery_labels)

    def recall(k, mode):
        topk = ranked[:, :k]
        if mode == "instance_dup":
            g = np.asarray(groups)
            hits = (g[topk] == g[targets, None]).any(axis=1)
        elif mode == "class":
            hits = (gallery_labels[topk] == query_labels[:, None]).any(axis=1)
        else:
            raise ValueError(mode)
        return float(hits.mean())

    out = {"class_R@1": recall(1, "class")}
    if groups is not None:
        out["instance_dup_R@1"] = recall(1, "instance_dup")
    return out


def within_class_dup_r1(
    query: np.ndarray,
    gallery: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Dup-aware R@1 retrieving only among same-class test rows; skip |class|<2."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    q = _l2_normalize(query)
    g = _l2_normalize(gallery)

    hits = []
    chances = []
    n_skipped_classes = 0
    evaluated_classes = set()

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            n_skipped_classes += 1
            continue
        evaluated_classes.add(int(c))
        # Restrict gallery to class members
        sims = q[idx] @ g[idx].T  # (nc, nc)
        # Top-1 among class (may retrieve self — true pair is same local index)
        pred_local = np.argmax(sims, axis=1)
        true_groups = groups[idx]
        pred_groups = true_groups[pred_local]
        hits.extend((pred_groups == true_groups).tolist())
        chances.extend([1.0 / len(idx)] * len(idx))

    return {
        "within_class_dup_R@1": float(np.mean(hits)) if hits else float("nan"),
        "chance": float(np.mean(chances)) if chances else float("nan"),
        "n_queries": len(hits),
        "n_classes_used": len(evaluated_classes),
        "n_classes_skipped": n_skipped_classes,
    }


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (centered features, Frobenius form)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xtx = X.T @ X
    yty = Y.T @ Y
    xty = X.T @ Y
    hsic_xy = float(np.sum(xty**2))
    hsic_xx = float(np.sum(xtx**2))
    hsic_yy = float(np.sum(yty**2))
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom <= 0:
        return 0.0
    return hsic_xy / denom


def mutual_knn_overlap(A: np.ndarray, B: np.ndarray, k: int = K_MUTUAL) -> float:
    """Mean fraction of shared kNN indices between spaces A and B (exclude self)."""
    n = A.shape[0]
    k_eff = min(k, n - 1)
    sim_a = cosine_sims(A, A)
    sim_b = cosine_sims(B, B)
    np.fill_diagonal(sim_a, -np.inf)
    np.fill_diagonal(sim_b, -np.inf)
    knn_a = np.argsort(-sim_a, axis=1)[:, :k_eff]
    knn_b = np.argsort(-sim_b, axis=1)[:, :k_eff]
    overlaps = []
    for i in range(n):
        overlaps.append(len(set(knn_a[i]).intersection(knn_b[i])) / k_eff)
    return float(np.mean(overlaps))


def probe_lr(X_train, Y_train, X_test, Y_test, seed: int) -> float:
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=3000, random_state=seed)
    clf.fit(scaler.transform(X_train), Y_train)
    return float(clf.score(scaler.transform(X_test), Y_test))


def center_by_train_t_mean(Z, gallery, train_t_mean):
    return Z - train_t_mean, gallery - train_t_mean


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def run_3a(I_tr, T_tr, Y_tr, I_te, T_te, Y_te, g_te, seed: int) -> dict:
    mu_t = class_means(T_tr, Y_tr)
    mu_i = class_means(I_tr, Y_tr)
    # Drop test rows whose class was absent in train
    train_classes = set(mu_t.keys())
    mask = np.array([int(y) in train_classes for y in Y_te])
    I_te, T_te, Y_te, g_te = I_te[mask], T_te[mask], Y_te[mask], g_te[mask]

    T_tr_r = apply_class_means(T_tr, Y_tr, mu_t)
    I_tr_r = apply_class_means(I_tr, Y_tr, mu_i)
    T_te_r = apply_class_means(T_te, Y_te, mu_t)
    I_te_r = apply_class_means(I_te, Y_te, mu_i)

    out = {}
    for name, fitter in [
        ("procrustes", lambda a, b: m02.fit_procrustes(a, b)),
        ("ridge", lambda a, b: m02.fit_ridge(a, b)),
    ]:
        predict = fitter(I_tr_r, T_tr_r)
        Z = np.asarray(predict(I_te_r), dtype=np.float64)
        metrics = within_class_dup_r1(Z, T_te_r, Y_te, g_te)
        out[name] = metrics
    return out


def run_3b(I_tr, T_tr, Y_tr, I_te, T_te, Y_te, seed: int) -> dict:
    predict = m02.fit_ridge(I_tr, T_tr)
    G_tr = np.asarray(predict(I_tr), dtype=np.float64)
    G_te = np.asarray(predict(I_te), dtype=np.float64)
    R_tr = T_tr.astype(np.float64) - G_tr
    R_te = T_te.astype(np.float64) - G_te

    rng = np.random.default_rng(seed)
    Y_tr_shuf = rng.permutation(Y_tr)

    features = {
        "I": (I_tr, I_te),
        "T": (T_tr, T_te),
        "R": (R_tr, R_te),
        "IR": (np.concatenate([I_tr, R_tr], axis=1), np.concatenate([I_te, R_te], axis=1)),
    }
    out = {}
    for name, (Xtr, Xte) in features.items():
        out[name] = probe_lr(Xtr, Y_tr, Xte, Y_te, seed)

    out["R_shuffled_labels"] = probe_lr(R_tr, Y_tr_shuf, R_te, Y_te, seed)
    return out


def run_3c(I_tr, T_tr, Y_tr, I_te, T_te, Y_te, g_te, seed: int) -> dict:
    train_t_mean = T_tr.mean(axis=0)
    out = {}
    for name, fitter in [
        ("procrustes", m02.fit_procrustes),
        ("ridge", m02.fit_ridge),
    ]:
        predict = fitter(I_tr, T_tr)
        Z = np.asarray(predict(I_te), dtype=np.float64)
        Zc, Gc = center_by_train_t_mean(Z, T_te.astype(np.float64), train_t_mean)

        cos = retrieval_from_sims(cosine_sims(Zc, Gc), Y_te, Y_te, groups=g_te)
        csls = retrieval_from_sims(csls_sims(Zc, Gc, k=K_CSLS), Y_te, Y_te, groups=g_te)
        out[name] = {"cosine": cos, "csls": csls}
    return out


def run_3d(I_tr, T_tr, Y_tr, I_te, T_te, Y_te, seed: int) -> dict:
    T_te64 = T_te.astype(np.float64)
    I_te64 = I_te.astype(np.float64)
    rng = np.random.default_rng(seed)
    I_shuf = I_te64[rng.permutation(len(I_te64))]

    out = {
        "mutual_knn_T_I": mutual_knn_overlap(T_te64, I_te64, k=K_MUTUAL),
        "mutual_knn_T_Ishuf": mutual_knn_overlap(T_te64, I_shuf, k=K_MUTUAL),
        "cka_T_I": linear_cka(T_te64, I_te64),
        "cka_T_Ishuf": linear_cka(T_te64, I_shuf),
        "maps": {},
    }
    for name, fitter in [
        ("procrustes", m02.fit_procrustes),
        ("ridge", m02.fit_ridge),
    ]:
        predict = fitter(I_tr, T_tr)
        G = np.asarray(predict(I_te), dtype=np.float64)
        out["maps"][name] = {
            "cka_gI_T": linear_cka(G, T_te64),
            "mutual_knn_gI_T": mutual_knn_overlap(G, T_te64, k=K_MUTUAL),
        }
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table_3a(summary: dict) -> None:
    print("\n=== 3a Class-residual within-class retrieval ===")
    print(f"{'method':<14s} {'dup R@1':>14s} {'chance':>14s} {'n_queries':>12s}")
    print("-" * 58)
    for method in ["procrustes", "ridge"]:
        r = summary[method]["within_class_dup_R@1"]
        c = summary[method]["chance"]
        n = summary[method]["n_queries"]
        print(
            f"{method:<14s} {r['mean']:.3f}±{r['std']:.3f}  "
            f"{c['mean']:.3f}±{c['std']:.3f}  {n['mean']:.0f}±{n['std']:.0f}"
        )


def print_table_3b(summary: dict) -> None:
    print("\n=== 3b Complementarity probes (test LR acc) ===")
    keys = ["I", "T", "R", "IR", "R_shuffled_labels"]
    print(f"{'feature':<22s} {'acc':>14s}")
    print("-" * 38)
    for k in keys:
        s = summary[k]
        print(f"{k:<22s} {s['mean']:.3f}±{s['std']:.3f}")


def print_table_3c(summary: dict) -> None:
    print("\n=== 3c CSLS vs cosine (centered, k=10) ===")
    print(
        f"{'method':<12s} {'sim':<8s} {'dup R@1':>14s} {'class R@1':>14s}"
    )
    print("-" * 52)
    for method in ["procrustes", "ridge"]:
        for sim in ["cosine", "csls"]:
            d = summary[method][sim]["instance_dup_R@1"]
            c = summary[method][sim]["class_R@1"]
            print(
                f"{method:<12s} {sim:<8s} "
                f"{d['mean']:.3f}±{d['std']:.3f}  {c['mean']:.3f}±{c['std']:.3f}"
            )


def print_table_3d(summary: dict) -> None:
    print("\n=== 3d Space agreement ===")
    print(f"{'pair':<22s} {'mutual-kNN':>14s} {'CKA':>14s}")
    print("-" * 52)
    print(
        f"{'T vs I':<22s} "
        f"{summary['mutual_knn_T_I']['mean']:.3f}±{summary['mutual_knn_T_I']['std']:.3f}  "
        f"{summary['cka_T_I']['mean']:.3f}±{summary['cka_T_I']['std']:.3f}"
    )
    print(
        f"{'T vs I (row-shuf)':<22s} "
        f"{summary['mutual_knn_T_Ishuf']['mean']:.3f}±{summary['mutual_knn_T_Ishuf']['std']:.3f}  "
        f"{summary['cka_T_Ishuf']['mean']:.3f}±{summary['cka_T_Ishuf']['std']:.3f}"
    )
    for method in ["procrustes", "ridge"]:
        mk = summary["maps"][method]["mutual_knn_gI_T"]
        ck = summary["maps"][method]["cka_gI_T"]
        print(
            f"{'g(I) vs T ('+method+')':<22s} "
            f"{mk['mean']:.3f}±{mk['std']:.3f}  {ck['mean']:.3f}±{ck['std']:.3f}"
        )


def plot_3a(summary, path):
    methods = ["procrustes", "ridge"]
    x = np.arange(len(methods))
    r1 = [summary[m]["within_class_dup_R@1"]["mean"] for m in methods]
    r1e = [summary[m]["within_class_dup_R@1"]["std"] for m in methods]
    ch = [summary[m]["chance"]["mean"] for m in methods]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x - 0.15, r1, 0.3, yerr=r1e, capsize=3, label="within-class dup R@1")
    ax.bar(x + 0.15, ch, 0.3, label="chance (mean 1/|class|)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Recall / chance")
    ax.set_title("3a Class-residual within-class retrieval")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_3b(summary, path):
    keys = ["I", "T", "R", "IR", "R_shuffled_labels"]
    means = [summary[k]["mean"] for k in keys]
    stds = [summary[k]["std"] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(len(keys)), means, yerr=stds, capsize=3)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("Test LR accuracy")
    ax.set_title("3b Complementarity probes")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_3c(summary, path):
    methods = ["procrustes", "ridge"]
    metrics = ["instance_dup_R@1", "class_R@1"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    x = np.arange(len(methods))
    w = 0.35
    for ax, metric in zip(axes, metrics):
        cos_m = [summary[m]["cosine"][metric]["mean"] for m in methods]
        cos_s = [summary[m]["cosine"][metric]["std"] for m in methods]
        csl_m = [summary[m]["csls"][metric]["mean"] for m in methods]
        csl_s = [summary[m]["csls"][metric]["std"] for m in methods]
        ax.bar(x - w / 2, cos_m, w, yerr=cos_s, capsize=3, label="cosine")
        ax.bar(x + w / 2, csl_m, w, yerr=csl_s, capsize=3, label="CSLS")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(metric)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    fig.suptitle("3c Centered cosine vs CSLS (k=10)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_3d(summary, path):
    labels = ["T–I", "T–Ishuf", "g–T proc", "g–T ridge"]
    knn = [
        summary["mutual_knn_T_I"]["mean"],
        summary["mutual_knn_T_Ishuf"]["mean"],
        summary["maps"]["procrustes"]["mutual_knn_gI_T"]["mean"],
        summary["maps"]["ridge"]["mutual_knn_gI_T"]["mean"],
    ]
    knn_e = [
        summary["mutual_knn_T_I"]["std"],
        summary["mutual_knn_T_Ishuf"]["std"],
        summary["maps"]["procrustes"]["mutual_knn_gI_T"]["std"],
        summary["maps"]["ridge"]["mutual_knn_gI_T"]["std"],
    ]
    cka = [
        summary["cka_T_I"]["mean"],
        summary["cka_T_Ishuf"]["mean"],
        summary["maps"]["procrustes"]["cka_gI_T"]["mean"],
        summary["maps"]["ridge"]["cka_gI_T"]["mean"],
    ]
    cka_e = [
        summary["cka_T_I"]["std"],
        summary["cka_T_Ishuf"]["std"],
        summary["maps"]["procrustes"]["cka_gI_T"]["std"],
        summary["maps"]["ridge"]["cka_gI_T"]["std"],
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(x, knn, yerr=knn_e, capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_title("Mutual kNN overlap (k=10)")
    axes[0].set_ylim(0, 1.05)
    axes[1].bar(x, cka, yerr=cka_e, capsize=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_title("Linear CKA")
    axes[1].set_ylim(0, 1.05)
    fig.suptitle("3d Space agreement")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    T, I, Y = load_data()
    groups = group_ids(T)
    print(f"Device helpers from 02_mapping; n={len(Y)}")

    store_3a = {
        m: {"within_class_dup_R@1": [], "chance": [], "n_queries": [], "n_classes_used": []}
        for m in ["procrustes", "ridge"]
    }
    store_3b = {k: [] for k in ["I", "T", "R", "IR", "R_shuffled_labels"]}
    store_3c = {
        m: {s: {"instance_dup_R@1": [], "class_R@1": []} for s in ["cosine", "csls"]}
        for m in ["procrustes", "ridge"]
    }
    store_3d = {
        "mutual_knn_T_I": [],
        "mutual_knn_T_Ishuf": [],
        "cka_T_I": [],
        "cka_T_Ishuf": [],
        "maps": {
            m: {"cka_gI_T": [], "mutual_knn_gI_T": []} for m in ["procrustes", "ridge"]
        },
    }
    per_seed = []

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        m02.set_seed(seed)
        train_idx, test_idx = group_split(Y, groups, test_size=0.2, seed=seed)
        I_tr, T_tr, Y_tr = I[train_idx], T[train_idx], Y[train_idx]
        I_te, T_te, Y_te = I[test_idx], T[test_idx], Y[test_idx]
        g_te = groups[test_idx]

        a = run_3a(I_tr, T_tr, Y_tr, I_te, T_te, Y_te, g_te, seed)
        b = run_3b(I_tr, T_tr, Y_tr, I_te, T_te, Y_te, seed)
        c = run_3c(I_tr, T_tr, Y_tr, I_te, T_te, Y_te, g_te, seed)
        d = run_3d(I_tr, T_tr, Y_tr, I_te, T_te, Y_te, seed)
        per_seed.append({"seed": seed, "3a": a, "3b": b, "3c": c, "3d": d})

        for m, met in a.items():
            store_3a[m]["within_class_dup_R@1"].append(met["within_class_dup_R@1"])
            store_3a[m]["chance"].append(met["chance"])
            store_3a[m]["n_queries"].append(met["n_queries"])
            store_3a[m]["n_classes_used"].append(met["n_classes_used"])
            print(
                f"  3a {m}: dupR@1={met['within_class_dup_R@1']:.3f} "
                f"chance={met['chance']:.3f} n={met['n_queries']}"
            )
        for k, v in b.items():
            store_3b[k].append(v)
        print(
            "  3b "
            + " ".join(f"{k}={b[k]:.3f}" for k in ["I", "T", "R", "IR", "R_shuffled_labels"])
        )
        for m in ["procrustes", "ridge"]:
            for s in ["cosine", "csls"]:
                store_3c[m][s]["instance_dup_R@1"].append(c[m][s]["instance_dup_R@1"])
                store_3c[m][s]["class_R@1"].append(c[m][s]["class_R@1"])
            print(
                f"  3c {m}: cos dup={c[m]['cosine']['instance_dup_R@1']:.3f} "
                f"csls dup={c[m]['csls']['instance_dup_R@1']:.3f} "
                f"cos cls={c[m]['cosine']['class_R@1']:.3f} "
                f"csls cls={c[m]['csls']['class_R@1']:.3f}"
            )
        for k in ["mutual_knn_T_I", "mutual_knn_T_Ishuf", "cka_T_I", "cka_T_Ishuf"]:
            store_3d[k].append(d[k])
        for m in ["procrustes", "ridge"]:
            store_3d["maps"][m]["cka_gI_T"].append(d["maps"][m]["cka_gI_T"])
            store_3d["maps"][m]["mutual_knn_gI_T"].append(d["maps"][m]["mutual_knn_gI_T"])
        print(
            f"  3d knn(T,I)={d['mutual_knn_T_I']:.3f} "
            f"knn(T,Ishuf)={d['mutual_knn_T_Ishuf']:.3f} "
            f"CKA(T,I)={d['cka_T_I']:.3f} "
            f"CKA(g_proc,T)={d['maps']['procrustes']['cka_gI_T']:.3f} "
            f"CKA(g_ridge,T)={d['maps']['ridge']['cka_gI_T']:.3f}"
        )

    summary_3a = {
        m: {k: mean_std(v) for k, v in store_3a[m].items()} for m in store_3a
    }
    summary_3b = {k: mean_std(v) for k, v in store_3b.items()}
    summary_3c = {
        m: {
            s: {k: mean_std(v) for k, v in store_3c[m][s].items()}
            for s in ["cosine", "csls"]
        }
        for m in store_3c
    }
    summary_3d = {
        k: mean_std(store_3d[k])
        for k in ["mutual_knn_T_I", "mutual_knn_T_Ishuf", "cka_T_I", "cka_T_Ishuf"]
    }
    summary_3d["maps"] = {
        m: {k: mean_std(v) for k, v in store_3d["maps"][m].items()}
        for m in ["procrustes", "ridge"]
    }

    payload = {
        "seeds": SEEDS,
        "3a": summary_3a,
        "3b": summary_3b,
        "3c": summary_3c,
        "3d": summary_3d,
        "per_seed": per_seed,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {RESULTS_PATH}")

    print_table_3a(summary_3a)
    print_table_3b(summary_3b)
    print_table_3c(summary_3c)
    print_table_3d(summary_3d)

    FIGS.mkdir(parents=True, exist_ok=True)
    plot_3a(summary_3a, FIGS / "03_a_residual.png")
    plot_3b(summary_3b, FIGS / "03_b_complementarity.png")
    plot_3c(summary_3c, FIGS / "03_c_csls.png")
    plot_3d(summary_3d, FIGS / "03_d_agreement.png")
    print(
        f"Wrote {FIGS / '03_a_residual.png'}, {FIGS / '03_b_complementarity.png'}, "
        f"{FIGS / '03_c_csls.png'}, {FIGS / '03_d_agreement.png'}"
    )


if __name__ == "__main__":
    main()

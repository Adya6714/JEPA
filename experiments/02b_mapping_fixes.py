"""Mapping fixes on top of 02_mapping: dup-aware R@K, centered cosine, hubness, InfoNCE."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import group_ids, load_data
from src.metrics import retrieval
from src.splits import group_split

# Reuse 02_mapping helpers
_spec = importlib.util.spec_from_file_location(
    "mapping02", ROOT / "experiments" / "02_mapping.py"
)
m02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m02)

SEEDS = list(range(5))
BASE_METHODS = ["procrustes", "ridge", "mlp_cos", "mlp_vicreg"]
INFONCE_TEMPS = [0.07, 0.2]
WEIGHT_DECAY = 1e-2
MODES = ["raw", "centered"]
RESULTS_PATH = ROOT / "results" / "02b.json"

TABLE_KEYS = ["instance_dup_R@1", "class_R@1", "effective_rank", "hub_skew"]


def method_names():
    names = list(BASE_METHODS)
    for t in INFONCE_TEMPS:
        names.append(f"infonce_t{t}")
    return names


def effective_rank(X: np.ndarray) -> float:
    """exp(entropy of normalized squared singular values) after centering."""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s.astype(np.float64) ** 2
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    p = p[p > 0]
    entropy = float(-(p * np.log(p)).sum())
    return float(np.exp(entropy))


def hubness(query: np.ndarray, gallery: np.ndarray, k: int = 10) -> dict:
    """How often each gallery row appears in someone's top-k; skewness + max."""
    from src.metrics import _l2_normalize

    q = _l2_normalize(query)
    g = _l2_normalize(gallery)
    sims = q @ g.T
    topk = np.argsort(-sims, axis=1)[:, :k]
    counts = np.bincount(topk.ravel(), minlength=gallery.shape[0]).astype(np.float64)
    return {
        "hub_skew": float(skew(counts, bias=False)),
        "hub_max": float(counts.max()),
        "hub_mean": float(counts.mean()),
    }


def apply_centering(Z: np.ndarray, gallery: np.ndarray, train_t_mean: np.ndarray, mode: str):
    if mode == "raw":
        return Z, gallery
    if mode == "centered":
        return Z - train_t_mean, gallery - train_t_mean
    raise ValueError(mode)


def evaluate_mode(
    Z_query: np.ndarray,
    T_gallery: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    train_t_mean: np.ndarray,
    mode: str,
) -> dict:
    Z, G = apply_centering(Z_query, T_gallery, train_t_mean, mode)
    ret = retrieval(Z, G, Y, Y, groups=groups)
    hubs = hubness(Z, G, k=10)
    return {
        "instance_R@1": ret["instance_R@1"],
        "instance_R@5": ret["instance_R@5"],
        "instance_dup_R@1": ret["instance_dup_R@1"],
        "instance_dup_R@5": ret["instance_dup_R@5"],
        "class_R@1": ret["class_R@1"],
        "class_R@5": ret["class_R@5"],
        "paired_cosine": m02.paired_cosine(Z, G),
        "effective_rank": effective_rank(Z_query),  # rank of predictions (uncentered geometry)
        **hubs,
    }


def infonce_loss_masked(pred, target, batch_groups, temperature: float):
    """Symmetric InfoNCE; same-group off-diagonal pairs are masked (not negatives)."""
    z_i = F.normalize(pred, dim=-1)
    z_t = F.normalize(target, dim=-1)
    logits = z_i @ z_t.T / temperature
    g = batch_groups
    same = g[:, None] == g[None, :]
    eye = torch.eye(g.shape[0], dtype=torch.bool, device=g.device)
    neg_mask = same & ~eye
    logits = logits.masked_fill(neg_mask, float("-inf"))
    labels = torch.arange(g.shape[0], device=g.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )


def fit_infonce_masked(I_train, T_train, groups_train, seed: int, temperature: float):
    """Linear InfoNCE with weight_decay, group-masked negatives, given temperature."""
    m02.set_seed(seed)
    device = m02.get_device()

    x_scaler = StandardScaler().fit(I_train)
    y_scaler = StandardScaler().fit(T_train)
    I_s = x_scaler.transform(I_train).astype(np.float32)
    T_s = y_scaler.transform(T_train).astype(np.float32)
    g_arr = np.asarray(groups_train, dtype=np.int64)

    ds = TensorDataset(
        torch.from_numpy(I_s),
        torch.from_numpy(T_s),
        torch.from_numpy(g_arr),
    )
    loader = DataLoader(ds, batch_size=m02.BATCH_SIZE, shuffle=True, drop_last=True)

    model = nn.Linear(512, 512, bias=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=m02.LR, weight_decay=WEIGHT_DECAY)

    model.train()
    for _ in range(m02.EPOCHS):
        for xb, tb, gb in loader:
            xb, tb, gb = xb.to(device), tb.to(device), gb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = infonce_loss_masked(pred, tb, gb, temperature)
            loss.backward()
            opt.step()

    model.eval()

    @torch.no_grad()
    def predict(I):
        Xs = x_scaler.transform(I).astype(np.float32)
        out = []
        for start in range(0, len(Xs), m02.BATCH_SIZE):
            batch = torch.from_numpy(Xs[start : start + m02.BATCH_SIZE]).to(device)
            out.append(model(batch).cpu().numpy())
        return y_scaler.inverse_transform(np.concatenate(out, axis=0))

    return predict


def print_table(summary: dict, methods: list[str]) -> None:
    header = f"{'method':<16s} {'mode':<10s}" + "".join(f"{k:>16s}" for k in TABLE_KEYS)
    print("\n" + header)
    print("-" * len(header))
    for method in methods:
        for mode in MODES:
            cells = []
            for key in TABLE_KEYS:
                m = summary[method][mode][key]["mean"]
                s = summary[method][mode][key]["std"]
                cells.append(f"{m:.3f}±{s:.3f}")
            print(f"{method:<16s} {mode:<10s}" + "".join(f"{c:>16s}" for c in cells))


def main():
    T, I, Y = load_data()
    groups = group_ids(T)
    methods = method_names()
    print(f"Device: {m02.get_device()}")
    print(f"Methods: {methods}")

    # Accumulate: summary structure built at end
    # raw_acc[method][mode][key] = list over seeds
    store = {m: {mode: {} for mode in MODES} for m in methods}
    for m in methods:
        for mode in MODES:
            for key in (
                TABLE_KEYS
                + ["instance_R@1", "instance_R@5", "instance_dup_R@5", "class_R@5",
                   "paired_cosine", "hub_max", "hub_mean"]
            ):
                store[m][mode][key] = []

    ref_erank = {"T_test": [], "I_test": []}
    infonce_train_r1 = {f"infonce_t{t}": {"raw": [], "centered": []} for t in INFONCE_TEMPS}
    per_seed = []

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        m02.set_seed(seed)
        train_idx, test_idx = group_split(Y, groups, test_size=0.2, seed=seed)
        I_tr, T_tr, Y_tr = I[train_idx], T[train_idx], Y[train_idx]
        I_te, T_te, Y_te = I[test_idx], T[test_idx], Y[test_idx]
        g_tr, g_te = groups[train_idx], groups[test_idx]
        train_t_mean = T_tr.mean(axis=0)

        ref_erank["T_test"].append(effective_rank(T_te.astype(np.float64)))
        ref_erank["I_test"].append(effective_rank(I_te.astype(np.float64)))
        print(
            f"  ref eff.rank T_test={ref_erank['T_test'][-1]:.2f} "
            f"I_test={ref_erank['I_test'][-1]:.2f}"
        )

        seed_row = {"seed": seed, "ref_effective_rank": {
            "T_test": ref_erank["T_test"][-1],
            "I_test": ref_erank["I_test"][-1],
        }}

        # Base methods from 02_mapping
        for method in BASE_METHODS:
            print(f"  fitting {method}...")
            predict = m02.FITTERS[method](I_tr, T_tr, seed)
            Z_te = np.asarray(predict(I_te), dtype=np.float64)
            seed_row[method] = {}
            for mode in MODES:
                metrics = evaluate_mode(Z_te, T_te.astype(np.float64), Y_te, g_te, train_t_mean, mode)
                seed_row[method][mode] = metrics
                for k, v in metrics.items():
                    if k in store[method][mode]:
                        store[method][mode][k].append(v)
                print(
                    f"    [{mode}] dupR@1={metrics['instance_dup_R@1']:.3f} "
                    f"clsR@1={metrics['class_R@1']:.3f} "
                    f"erank={metrics['effective_rank']:.2f} "
                    f"hub_skew={metrics['hub_skew']:.2f}"
                )

        # InfoNCE variants
        for temp in INFONCE_TEMPS:
            name = f"infonce_t{temp}"
            print(f"  fitting {name} (wd={WEIGHT_DECAY}, group-masked)...")
            predict = fit_infonce_masked(I_tr, T_tr, g_tr, seed, temperature=temp)
            Z_tr = np.asarray(predict(I_tr), dtype=np.float64)
            Z_te = np.asarray(predict(I_te), dtype=np.float64)
            seed_row[name] = {}
            for mode in MODES:
                test_m = evaluate_mode(Z_te, T_te.astype(np.float64), Y_te, g_te, train_t_mean, mode)
                train_m = evaluate_mode(Z_tr, T_tr.astype(np.float64), Y_tr, g_tr, train_t_mean, mode)
                test_m["train_instance_dup_R@1"] = train_m["instance_dup_R@1"]
                test_m["train_instance_R@1"] = train_m["instance_R@1"]
                seed_row[name][mode] = test_m
                infonce_train_r1[name][mode].append(train_m["instance_dup_R@1"])
                for k, v in test_m.items():
                    if k in store[name][mode]:
                        store[name][mode][k].append(v)
                print(
                    f"    [{mode}] train_dupR@1={train_m['instance_dup_R@1']:.3f} "
                    f"test_dupR@1={test_m['instance_dup_R@1']:.3f} "
                    f"clsR@1={test_m['class_R@1']:.3f} "
                    f"erank={test_m['effective_rank']:.2f} "
                    f"hub_skew={test_m['hub_skew']:.2f}"
                )

        per_seed.append(seed_row)

    # Aggregate
    summary = {}
    for method in methods:
        summary[method] = {}
        for mode in MODES:
            summary[method][mode] = {}
            for key, vals in store[method][mode].items():
                if not vals:
                    continue
                arr = np.asarray(vals, dtype=np.float64)
                summary[method][mode][key] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "per_seed": arr.tolist(),
                }

    ref_summary = {
        k: {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "per_seed": v,
        }
        for k, v in ref_erank.items()
    }

    infonce_train_summary = {}
    for name, modes in infonce_train_r1.items():
        infonce_train_summary[name] = {}
        for mode, vals in modes.items():
            arr = np.asarray(vals, dtype=np.float64)
            infonce_train_summary[name][mode] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "per_seed": arr.tolist(),
            }

    payload = {
        "seeds": SEEDS,
        "methods": methods,
        "modes": MODES,
        "ref_effective_rank": ref_summary,
        "infonce_train_dup_R@1": infonce_train_summary,
        "summary": summary,
        "per_seed": per_seed,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {RESULTS_PATH}")

    print(
        f"\nReference effective rank — "
        f"T_test: {ref_summary['T_test']['mean']:.2f}±{ref_summary['T_test']['std']:.2f}, "
        f"I_test: {ref_summary['I_test']['mean']:.2f}±{ref_summary['I_test']['std']:.2f}"
    )
    print_table(summary, methods)

    print("\nInfoNCE train dup-aware R@1 (raw):")
    for name in [f"infonce_t{t}" for t in INFONCE_TEMPS]:
        s = infonce_train_summary[name]["raw"]
        te = summary[name]["raw"]["instance_dup_R@1"]
        print(
            f"  {name}: train {s['mean']:.3f}±{s['std']:.3f}  "
            f"test {te['mean']:.3f}±{te['std']:.3f}"
        )


if __name__ == "__main__":
    main()

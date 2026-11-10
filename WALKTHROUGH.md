# End-to-end walkthrough: JEPA embedding alignment study

This document explains **what this repo is**, **what question it answers**, **how every experiment works**, and **what to conclude**. Read this first; use [`FINDINGS.md`](FINDINGS.md) as the short research log with claim statuses.

---

## 1. One-paragraph summary

We have paired **image embeddings** `I` and **text embeddings** `T` (2500 rows × 512 dims, 31 classes). The research asks: *how aligned are these spaces, how should we map `I → T`, and does `T` add information beyond `I`?*  

We discovered that naive evaluation is misleading (train/test leakage via duplicate `T`s; raw cosine dominated by a shared mean / hub direction). After fixing those, **linear maps (Procrustes / Ridge) are enough for class-level alignment**, **InfoNCE overfits**, **`T` is mostly class-level**, and **`g_ridge(I)` helps few-shot classification more than raw `I` at k=1**.

---

## 2. Data

| Array | Shape | Meaning |
|-------|-------|---------|
| `T` | `(2500, 512)` float32 | Text / caption embedding per row |
| `I` | `(2500, 512)` float32 | Image embedding paired with that row |
| `Y` | `(2500,)` int64 | Class label (31 classes, ids 1…31) |

File: [`data/embeddings.npz`](data/embeddings.npz). Loader: [`src/data.py`](src/data.py) → `load_data()`.

**Duplicate texts.** Many rows share (nearly) the same `T` vector. `group_ids(T)` rounds to 6 decimals and assigns one integer id per unique text. There are **2075 groups** for 2500 rows — so ~425 rows are text duplicates. That matters for train/test splits and for “instance” retrieval.

---

## 3. Shared protocol (apply everywhere)

These rules were fixed early and reused in every later experiment:

1. **Group-aware splits** — `StratifiedGroupKFold` so no duplicate-`T` group appears in both train and test ([`src/splits.py`](src/splits.py) `group_split`).
2. **Seeds `0..4`** — report **mean ± std**.
3. **Fit scalers / means on train only** — never peek at test.
4. **Primary retrieval metric uses train-`T`-mean centering** — subtract the train mean of `T` from both predictions and the `T` gallery before L2-normalize + cosine ([`FINDINGS.md`](FINDINGS.md) decision 2026-10-13). Raw (uncentered) cosine is diagnostic only.

Global seed constant: `SEED = 0` in [`src/__init__.py`](src/__init__.py) (default for helpers; each experiment still loops seeds explicitly).

### How to run anything

```bash
cd /path/to/JEPA
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
PYTHONPATH=. python experiments/01_leakage_audit.py
# … same pattern for 02, 02b, 03, 04
PYTHONPATH=. python experiments/make_report_figs.py  # figs/final/
```

Outputs always go to `results/*.json` and `figs/`. Publication panels: `figs/final/F1`–`F4`.

---

## 4. Code map

```
JEPA/
├── data/embeddings.npz          # T, I, Y
├── src/
│   ├── data.py                  # load_data, group_ids
│   ├── splits.py                # group_split, random_split
│   └── metrics.py               # retrieval (+ dup-aware instance R@K)
├── experiments/
│   ├── 01_leakage_audit.py      # random vs group probes
│   ├── 02_mapping.py            # I→T maps (first pass; some metrics flawed)
│   ├── 02b_mapping_fixes.py     # corrected retrieval / hubness / InfoNCE
│   ├── 03_probes.py             # residuals, complementarity, CSLS, CKA
│   ├── 04_fewshot.py            # k-shot LR on I / T / g(I)
│   └── make_report_figs.py      # publication figures
├── results/                     # JSON numbers
├── figs/                        # plots (+ figs/final/)
├── FINDINGS.md                  # decisions + verified claims
└── WALKTHROUGH.md               # this file
```

---

## 5. Experiment-by-experiment

### 5.1 Experiment 01 — Leakage audit

**Script:** [`experiments/01_leakage_audit.py`](experiments/01_leakage_audit.py)  
**Outputs:** [`results/01_leakage.json`](results/01_leakage.json), [`figs/01_leakage.png`](figs/01_leakage.png), [`figs/final/F1_leakage.png`](figs/final/F1_leakage.png)

**Question:** If we randomly split rows, do duplicate texts put the “same” text in both train and test? Does that inflate classifier accuracy?

**Method:**

- For seeds 0–4, split with either `random_split` or `group_split`.
- Features: `T`, `I`, `concat[T,I]`.
- Models: `LogisticRegression(max_iter=3000)`, `KNeighborsClassifier(k=5, metric="cosine")`.
- `StandardScaler` fit on train only; report **test accuracy**.
- On random splits only: fraction of test rows whose **exact** `T` vector also appears in train.

**Results (group split, LR):**

| Feature | Accuracy |
|---------|----------|
| `T` | 0.856 ± 0.009 |
| `I` | 0.948 ± 0.006 |
| `TI` | 0.974 ± 0.005 |

Random-split exact-`T` leak: **21.4% ± 2.1%**. Accuracy drop random→group ≤ **1.2 pts** (worst: T × kNN).

**Takeaway:** Leakage is real; probe conclusions (`I` ≫ `T`, concat helps a bit) are **robust**. Still use group splits going forward so retrieval/mapping aren’t contaminated.

---

### 5.2 Experiment 02 — Learn `g: I → T` (first pass)

**Script:** [`experiments/02_mapping.py`](experiments/02_mapping.py)  
**Outputs:** [`results/02_mapping.json`](results/02_mapping.json), [`figs/02_mapping.png`](figs/02_mapping.png)

**Question:** Which objective learns a good map from image space into text space?

**Methods fitted on train (group split):**

| Name | Idea |
|------|------|
| **procrustes** | Center `I`,`T`; orthogonal `R` via `scipy.linalg.orthogonal_procrustes`; `g(I)=(I−μ_I)R+μ_T` |
| **ridge** | `StandardScaler` on `I` and `T`; `RidgeCV` alphas `{0.1,1,10,100,1000}`, 5-fold CV |
| **mlp_cos** | MLP `512→1024→512` GELU; loss `1 − cos(g(I),T)`; AdamW 1e-3, 200 epochs, batch 256 |
| **mlp_vicreg** | Same MLP; cosine + variance hinge + off-diagonal cov penalty |
| **infonce** | Linear `W`; batch symmetric InfoNCE, temperature 0.07 |

**Test metrics (original 02):** paired cosine, MSE, instance/class R@1/R@5, “effective rank”, linear probe on `g(I)`.

**Known bugs in 02 (fixed in 02b):**

1. Effective rank reported **entropy** of singular-value spectrum, not **`exp(entropy)`**.
2. Instance retrieval required the **exact row index**, not “same duplicate-`T` group”.
3. Raw cosine without centering → hubness artifact (Procrustes looked best at retrieval for the wrong reason).
4. InfoNCE looked weak on test but train R@1 wasn’t reported → overfitting hidden.

Treat **02 numbers as historical**; quote **02b** for retrieval / rank / InfoNCE.

---

### 5.3 Experiment 02b — Mapping fixes (canonical mapping results)

**Script:** [`experiments/02b_mapping_fixes.py`](experiments/02b_mapping_fixes.py)  
**Outputs:** [`results/02b.json`](results/02b.json), [`figs/final/F2_hubness.png`](figs/final/F2_hubness.png)

**Fixes:**

1. **Effective rank** = `exp(H)` where `H` is entropy of normalized squared singular values after centering. Also report for raw `T_test` and `I_test` (~58).
2. **Dup-aware instance R@K** in [`src/metrics.py`](src/metrics.py): a hit if any top-K gallery row shares the **true pair’s group id**.
3. Evaluate every method under **raw** and **centered** cosine (subtract train `T` mean from predictions and gallery).
4. InfoNCE: `weight_decay=1e-2`, mask same-group in-batch negatives, temps `{0.07, 0.2}`, report **train and test** R@1.
5. **Hubness:** how often each gallery row appears in someone’s top-10; report skewness and max count.

**Headline numbers (centered, means over 5 seeds):**

| Method | dup R@1 | class R@1 | eff. rank | hub skew |
|--------|--------:|----------:|----------:|---------:|
| procrustes | 0.160 | 0.813 | 58.3 | 0.81 |
| ridge | 0.147 | 0.811 | 17.9 | 0.86 |
| mlp_vicreg | 0.130 | 0.735 | 65.7 | 0.67 |
| mlp_cos | 0.120 | 0.709 | 57.2 | 0.95 |
| infonce_t0.2 | 0.147 | 0.791 | 57.8 | 0.50 |
| infonce_t0.07 | 0.128 | 0.726 | 104.1 | 0.46 |

**InfoNCE overfitting (centered dup R@1):**  
t=0.07 train **0.905** vs test **0.128**; t=0.2 train **0.618** vs test **0.147**.

**Hubness story:** raw hub-skew order **exactly inverts** raw class R@1 across 6 methods. Centering collapses all hub skews below 1. So “who wins retrieval under raw cosine” was largely “who doesn’t collapse onto the mean hub.”

**Sanity:** Procrustes eff. rank = `I_test` eff. rank (orthogonal map preserves spectrum).

---

### 5.4 Experiment 03 — Deeper probes

**Script:** [`experiments/03_probes.py`](experiments/03_probes.py)  
**Outputs:** [`results/03_probes.json`](results/03_probes.json), `figs/03_*.png`, [`figs/final/F3_complementarity.png`](figs/final/F3_complementarity.png)

#### 3a — Class-residual / within-class retrieval

Remove class means (fit on train) from `T` and `I`, map residuals with Procrustes/Ridge, then retrieve **only among same-class test rows**. Report dup-aware within-class R@1 vs chance `mean(1/|class|)`.

| Method | Within-class dup R@1 | Chance |
|--------|---------------------:|-------:|
| procrustes | 0.197 ± 0.013 | 0.063 |
| ridge | 0.189 ± 0.007 | 0.063 |

Above chance (~3×), but still modest → some instance structure remains after removing class means, not a lot.

#### 3b — Complementarity of residual `R = T − g_ridge(I)`

Fit Ridge `g: I→T` on full train. Probe LR test accuracy:

| Feature | Acc |
|---------|----:|
| `I` | 0.948 |
| `T` | 0.856 |
| `R` | 0.460 |
| `concat[I,R]` | 0.973 |
| `R` + shuffled train labels | 0.031 |

`R` is far above chance (0.031) → **text residual still carries class signal** that linear `g(I)` didn’t capture. `concat[I,R]` ≈ earlier `concat[I,T]` gain.

#### 3c — CSLS vs cosine (centered, k=10)

Small lift in dup R@1 (e.g. Procrustes 0.160 → 0.175); class R@1 essentially flat. Hubness fix via centering already did most of the work; CSLS is a minor extra.

#### 3d — Space agreement

| Pair | Mutual kNN (k=10) | Linear CKA |
|------|------------------:|-----------:|
| `T` vs `I` | 0.255 | 0.387 |
| `T` vs row-shuffled `I` | 0.021 | 0.049 |
| `g_proc(I)` vs `T` | 0.255 | 0.387 |
| `g_ridge(I)` vs `T` | 0.286 | 0.470 |

Shuffled controls ≈ floor. Procrustes CKA matches `T`–`I` (orthogonal invariance of linear CKA). Ridge aligns geometry more tightly to `T`.

---

### 5.5 Experiment 04 — Few-shot probes

**Script:** [`experiments/04_fewshot.py`](experiments/04_fewshot.py)  
**Outputs:** [`results/04_fewshot.json`](results/04_fewshot.json), [`figs/04_fewshot.png`](figs/04_fewshot.png), [`figs/final/F4_fewshot.png`](figs/final/F4_fewshot.png)

**Setup:**

- Seeds 0–4 group splits; **10 random label draws** per split → **50 runs**.
- Fit Procrustes + Ridge on **all** train pairs.
- Features (T and g(*) centered by train `T` mean): `I`, `T`, `g_proc(I)`, `g_ridge(I)`, `concat[I, g_ridge(I)]`.
- For `k ∈ {1,5,10}`: sample **k examples per class** from train (k×31 rows), fit `StandardScaler` + `LogisticRegression` on those only; evaluate on **full test**.

| Feature | k=1 | k=5 | k=10 |
|---------|----:|----:|-----:|
| `I` | 0.650 ± 0.032 | 0.879 ± 0.012 | 0.915 ± 0.014 |
| `T` | 0.469 ± 0.045 | 0.754 ± 0.023 | 0.802 ± 0.015 |
| `g_proc` | 0.615 ± 0.031 | 0.865 ± 0.016 | 0.908 ± 0.015 |
| `g_ridge` | **0.784 ± 0.031** | **0.908 ± 0.013** | 0.927 ± 0.011 |
| `I + g_ridge` | 0.746 ± 0.031 | 0.907 ± 0.011 | **0.931 ± 0.011** |

**Takeaway:** Mapping image features into text space with Ridge **helps most when labels are scarce** (k=1: +13 pts over raw `I`). At k=10 the gap shrinks; concat still edges out.

---

## 6. Metrics glossary

| Term | Meaning |
|------|---------|
| **Paired cosine** | Mean cosine between row `i` of predictions and row `i` of `T` |
| **Instance R@1** | Is gallery row `i` the top-1 neighbor of query `i`? |
| **Dup-aware instance R@1** | Is any top-1 (or top-K) gallery row in the **same `T`-duplicate group** as the true pair? |
| **Class R@1** | Does top-1 gallery share the query’s class label? |
| **Centered cosine** | Subtract train mean of `T` from query predictions and gallery `T`, then L2-normalize |
| **Hub skew** | Skewness of how often each gallery item appears in someone’s top-10 |
| **Effective rank** | `exp(−∑ p log p)` with `p ∝ σ²` of centered matrix (Roy–Vetterli) |
| **CSLS** | `2·cos / (r_q + r_g)` with `r` = similarity to k-th NN in the other set |
| **Linear CKA** | Kernel alignment with linear kernels on centered features |
| **Mutual kNN** | Average fraction of shared k-nearest neighbor indices between two spaces |

---

## 7. What we believe (executive conclusions)

1. **Always group-split** on duplicate texts; random splits leak ~21% of test texts into train (accuracy barely moves for probes, but it’s the right protocol).
2. **Never trust raw cosine** for this dataset without centering — hubness from a shared mean direction invents fake method rankings.
3. **Linear maps suffice** for class-level `I→T` alignment; deep InfoNCE overfits train retrieval.
4. **`I` is a better class probe than `T`**, but **`T − g(I)` still has class signal**, and concat helps.
5. **`T` is mostly class-level** (class R@1 ~0.81 vs instance ~0.16); within-class residual retrieval is only ~0.20 vs chance ~0.06.
6. **Practical win:** use **`g_ridge(I)`** (or concat with `I`) for **few-shot** classification — big gain at k=1.

---

## 8. Open questions (still open)

1. Matched-dimension control: is concat better than `I` only because of width (1024 vs 512)?
2. Does raw `T` contain instance identity that no `I→T` map preserves?
3. What semantic content lives in residual `R = T − g(I)` (qualitative / retrieval examples)?

---

## 9. Suggested reading order

1. This file (overview).
2. [`FINDINGS.md`](FINDINGS.md) (claim statuses).
3. Figures in [`figs/final/`](figs/final/) (F1–F4).
4. JSON under [`results/`](results/) when you need exact per-seed numbers.
5. Source in order: `01` → `02` → `02b` → `03` → `04`.

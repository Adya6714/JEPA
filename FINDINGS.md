# Decisions log

| Date | Decision | Why | Alternatives rejected |
|------|----------|-----|----------------------|
| 2026-10-12 | Use group-aware splits (`StratifiedGroupKFold` on duplicate-`T` groups) for all main analyses | Random splits leak: **21.4%±2.1%** of test rows have an exact `T` twin in train (`results/01_leakage.json`) | Plain stratified `train_test_split` as the primary protocol (kept only as a leakage comparison) |
| 2026-10-12 | Run seeds `0..4` and report **mean±std** | Quantify split/optimizer variance without a full nested CV | Single seed; median-only reporting |
| 2026-10-12 | Fit all scalers / centering statistics on **train only** | Prevent test-set leakage into preprocessing | Fit on full data; fit on train+test |
| 2026-10-13 | Report all retrieval with **train-`T`-mean centering as primary**, raw cosine as secondary | Raw cosine rankings are hub-dominated by a shared mean direction; after centering, Procrustes and Ridge tie and hub skews collapse (`results/02b.json`) | Raw cosine as the primary retrieval protocol |
| 2026-10-13 | Prefer linear maps (Ridge / Procrustes) over InfoNCE for `I→T` | InfoNCE overfits train retrieval; linear maps match class R@1 after centering (`results/02b.json`) | InfoNCE / MLP cosine as default map |

# Implementation log

| Script | What it does | Outputs | Known issues |
|--------|--------------|---------|--------------|
| `experiments/01_leakage_audit.py` | Random vs group splits for LR / kNN on `T`, `I`, `concat[T,I]`; exact-`T` leak fraction | `results/01_leakage.json`, `figs/01_leakage.png`, `figs/final/F1_leakage.png` | — |
| `experiments/02_mapping.py` | `g: I → T` (Procrustes, Ridge, MLP cosine, MLP VICReg, InfoNCE) | `results/02_mapping.json`, `figs/02_mapping.png` | Eff. rank = entropy not `exp(H)`; instance R@K not dup-aware; raw cosine hub-biased. Superseded by 02b for those metrics. |
| `experiments/02b_mapping_fixes.py` | Corrected rank, dup-aware R@K, raw vs centered, hubness, InfoNCE wd + group mask + temps | `results/02b.json`, `figs/final/F2_hubness.png` | Canonical mapping / retrieval numbers |
| `experiments/03_probes.py` | Within-class residual retrieval; `R=T−g(I)` complementarity; CSLS; CKA / mutual-kNN | `results/03_probes.json`, `figs/03_*.png`, `figs/final/F3_complementarity.png` | — |
| `experiments/04_fewshot.py` | k-shot LR on `I`/`T`/`g(I)` (50 runs) | `results/04_fewshot.json`, `figs/04_fewshot.png`, `figs/final/F4_fewshot.png` | — |
| `experiments/make_report_figs.py` | Publication figures F1–F4 | `figs/final/` | Needs prior JSON results |

Shared library: `src/data.py`, `src/splits.py`, `src/metrics.py`. Narrative guide: `WALKTHROUGH.md`.

# Findings

### VERIFIED — Leakage exists but barely moves probe accuracy
- **Claim:** Exact-`T` leakage under random splits is ~21%, yet group-aware splits change test accuracy by ≤1.2 pts.
- **Evidence:** Leak **0.214±0.021**; worst drop T×kNN **0.792→0.780** (`results/01_leakage.json`).
- **Status:** VERIFIED

### VERIFIED — `I` beats `T` for class probing
- **Evidence:** Group-split LR **I: 0.948±0.006** vs **T: 0.856±0.009** (`results/01_leakage.json`).
- **Status:** VERIFIED

### TENTATIVE — `concat[T,I]` / `concat[I,R]` beat `I` alone
- **Evidence:** LR **TI: 0.974** (`01`); **IR: 0.973** with `R=T−g_ridge(I)` (`03` 3b).
- **Caveat:** No matched-dimension (512) control yet.
- **Status:** TENTATIVE

### RETRACTED — “Ridge wins geometry, Procrustes wins retrieval” (raw 02)
- **Why:** Raw-cosine ranking was mean/hub-driven; after centering they tie (`results/02b.json`).
- **Status:** RETRACTED

### VERIFIED — After centering, Procrustes and Ridge tie on retrieval
- **Evidence:** Centered class R@1 **0.813±0.030** vs **0.811±0.012**; dup R@1 **0.160±0.021** vs **0.147±0.019** (`results/02b.json`).
- **Status:** VERIFIED

### VERIFIED — Raw hub skew exactly inverts raw class R@1 (n=6)
- **Evidence:** Order proc→…→infonce_t0.07 for class R@1 is reverse of hub skew; all centered hub skews **< 1** (`results/02b.json`).
- **Status:** VERIFIED

### VERIFIED — Raw paired cosine is inflated by the mean
- **Evidence:** Centered paired cosine Ridge **0.51**, Procrustes **0.39**, InfoNCE t0.07 **0.26** (raw ~0.89 / 0.80 / 0.86) (`results/02b.json`).
- **Status:** VERIFIED

### VERIFIED — InfoNCE overfits
- **Evidence:** Centered dup R@1 train vs test — t0.07 **0.905 vs 0.128**; t0.2 **0.618 vs 0.147** (`results/02b.json`).
- **Status:** VERIFIED

### VERIFIED — Procrustes eff. rank equals `I_test`
- **Evidence:** Both **58.323±0.882** (`results/02b.json`).
- **Status:** VERIFIED

### VERIFIED — Residual `R = T − g_ridge(I)` still carries class signal
- **Claim:** Linear maps do not exhaust class information in `T`.
- **Evidence:** LR on `R`: **0.460±0.017** vs shuffled-label control **0.031±0.011**; `concat[I,R]` **0.973±0.005** (`results/03_probes.json` → `3b`).
- **Status:** VERIFIED

### VERIFIED — Within-class residual retrieval is above chance but modest
- **Evidence:** After removing train class means, within-class dup R@1 Procrustes **0.197±0.013** vs chance **0.063±0.002** (`results/03_probes.json` → `3a`).
- **Status:** VERIFIED

### VERIFIED — `T` and `I` spaces agree above shuffle controls
- **Evidence:** Mutual kNN **0.255** / CKA **0.387** vs shuffled **0.021** / **0.049**; Ridge `g(I)`–`T` CKA **0.470** (`results/03_probes.json` → `3d`).
- **Status:** VERIFIED

### VERIFIED — CSLS gives only a small centered retrieval lift
- **Evidence:** Procrustes dup R@1 cosine **0.160** → CSLS **0.175**; class R@1 flat (`results/03_probes.json` → `3c`).
- **Status:** VERIFIED

### VERIFIED — `g_ridge(I)` helps few-shot classification (especially k=1)
- **Evidence:** Over 50 runs, k=1 accuracy **g_ridge 0.784±0.031** vs **I 0.650±0.032**; k=10 concat edges **0.931** (`results/04_fewshot.json`).
- **Status:** VERIFIED

### TENTATIVE — `T` encodes mostly class-level information
- **Evidence:** Best centered class R@1 ~**0.81** vs instance ~**0.16**; within-class residual ~**0.20** vs chance ~**0.06**.
- **Caveat:** Does not fully separate “weak instance signal in `T`” from “maps destroy instance signal.”
- **Status:** TENTATIVE

### RETRACTED — Effective-rank numbers from experiment 02
- **Evidence of error:** 02 returned entropy, not `exp(entropy)`. Use `results/02b.json`.
- **Status:** RETRACTED

# Open questions

1. **Is concat better than `I` beyond dimensionality?** Need PCA-to-512 or width-matched controls.
2. **What semantics live in `R = T − g(I)`?** Qualitative nearest-neighbor / caption inspection.
3. **Can a nonlinear map capture residual class signal without InfoNCE-style overfitting?**

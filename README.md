# JEPA embeddings research

Paired image (`I`) and text (`T`) embeddings (2500×512, 31 classes): measure alignment, learn `I → T` maps, and test whether `T` adds signal beyond `I`.

**Start here:** [`WALKTHROUGH.md`](WALKTHROUGH.md) — full end-to-end explanation (data → methods → results → conclusions).  
**Research log:** [`FINDINGS.md`](FINDINGS.md) — decisions, claim statuses (VERIFIED / TENTATIVE / RETRACTED).

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=. python experiments/01_leakage_audit.py   # or 02 / 02b / 03 / 04
PYTHONPATH=. python experiments/make_report_figs.py    # → figs/final/
```

## Layout

| Path | Role |
|------|------|
| `data/` | `embeddings.npz` with arrays `T`, `I`, `Y` |
| `src/` | `load_data`, `group_ids`, splits, retrieval metrics |
| `experiments/` | One script per study (`01`…`04`, `make_report_figs`) |
| `results/` | JSON metrics |
| `figs/` | Plots; publication panels in `figs/final/` (`F1`–`F4`) |

## Pipeline (short)

1. **01** — Random vs group splits: ~21% exact-`T` leak; probes barely change → use group splits.
2. **02 / 02b** — Map `I → T`; raw cosine is hub-dominated; after centering, Procrustes ≈ Ridge; InfoNCE overfits.
3. **03** — Residuals, complementarity (`R = T − g(I)`), CSLS, CKA / mutual-kNN.
4. **04** — Few-shot LR: `g_ridge(I)` beats raw `I` at k=1 (0.78 vs 0.65).

Global seed default: `0` (`src/SEED`). All scalers / means fit on **train only**.

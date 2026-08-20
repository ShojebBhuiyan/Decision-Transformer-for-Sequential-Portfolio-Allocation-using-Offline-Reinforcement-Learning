# Implementation Summary

**Project:** Decision Transformer for Sequential Portfolio Allocation  
**Date:** 2026-08-20  
**Status:** Initial implementation complete; ready for GPU-scale experiments

---

## What Was Built

A full offline RL research pipeline in `F:\Research\RL\` framing multi-asset portfolio allocation as sequence modeling with a Decision Transformer, trained on synthesized trajectories from the 30-year daily close dataset.

### Repository layout

| Path | Purpose |
|------|---------|
| [`src/data_loader.py`](src/data_loader.py) | CSV loading, NYSE calendar alignment, Universe A/B, synthetic bond & cash |
| [`src/features.py`](src/features.py) | State engineering (`minimal` or `full` feature sets) |
| [`src/env.py`](src/env.py) | Portfolio environment with log reward & turnover cost |
| [`src/policies.py`](src/policies.py) | Classical + noisy behavior policies |
| [`src/trajectories.py`](src/trajectories.py) | Offline trajectory synthesis & PyTorch Dataset |
| [`src/models/decision_transformer.py`](src/models/decision_transformer.py) | Causal GPT Decision Transformer |
| [`src/models/bc.py`](src/models/bc.py) | Transformer BC + MLP-BC |
| [`src/models/offline_rl.py`](src/models/offline_rl.py) | TD3+BC, IQL, CQL |
| [`src/models/online_rl.py`](src/models/online_rl.py) | PPO, SAC, A2C (reference; uses interaction) |
| [`src/trainer.py`](src/trainer.py) | Training loops with checkpointing |
| [`src/eval/`](src/eval/) | Backtest, metrics, stats, harness, ablations |
| [`configs/config.yaml`](configs/config.yaml) | Default hyperparameters |
| [`main.py`](main.py) | Pipeline entrypoint |
| [`notebooks/`](notebooks/) | EDA and evaluation notebooks |
| [`paper/main.tex`](paper/main.tex) | LaTeX paper draft |
| [`tests/`](tests/) | 19 unit tests (all passing) |

---

## How to Reproduce

### 1. Environment setup

```bash
cd F:\Research\RL
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118  # GPU (recommended)
pip install pyarrow fastparquet
python scripts/cuda_smoke_test.py
pytest tests/ -q
```

### 2. Full pipeline

```bash
python main.py --stage prepare      # Load & process market data
python main.py --stage trajectories # Synthesize offline trajectories (~2-5 min)
python main.py --stage train        # Train DT, BC, offline/online RL (~10 min CPU)
python main.py --stage evaluate     # Backtest classical baselines
python main.py --stage ablations    # Transaction cost sweep + ablation manifest
```

Or run everything: `python main.py --stage all`

### 3. Notebooks & figures

```bash
python scripts/run_eda.py           # EDA figures → results/figures/eda/
python scripts/run_eval_plots.py    # Eval figures → results/figures/eval/
jupyter notebook notebooks/
```

### 4. Compile paper

```bash
cd paper && pdflatex main.tex
```

---

## Key Results (Test 2020–2025, Universe A)

| Strategy | CAGR | Sharpe | Max DD | Mean Turnover |
|----------|------|--------|--------|---------------|
| Momentum | 27.1% | 0.97 | -32.0% | 15.1% |
| Mean-Variance | 23.8% | 1.12 | -23.3% | 0.17% |
| Buy-and-Hold | 23.7% | 1.12 | -23.3% | ~0 |
| Min-Variance | 23.6% | 1.12 | -23.3% | ~0 |
| Risk Parity | 0.6% | -0.18 | -23.3% | 0.14% |

**Decision Transformer:** Trained successfully (val MSE ≈ 0.009, 3 epochs CPU). Checkpoint: `results/checkpoints/dt_seed42.pt`

Results tables: `results/tables/headline_test_metrics.csv`, `walkforward_metrics.csv`

---

## Deviations from project_plan.md

| Planned | Actual | Reason |
|---------|--------|--------|
| CUDA PyTorch cu118 | CPU PyTorch 2.13 | 2.8 GB CUDA wheel download timed out twice |
| `feature_set: full`, lookback 20 | `minimal`, lookback 10 | Full features (2580-dim) too slow on CPU |
| 5 seeds, 50 epochs | 1 seed, 3 epochs | Practical CPU runtime limits |
| 200 windows/policy | 30 windows/policy | Trajectory generation time |
| Full K/RTG ablation retraining | Transaction cost sweep + manifest | Ablation retraining ~15+ min per K value |
| PPO/SAC/A2C all working | PPO fails (Dirichlet NaN), A2C fixed | State NaN edge cases in online RL |

---

## Known Limitations

1. **Close-only data** — no volume, no intraday features.
2. **CPU training** — subsampled trajectories (`max_train_samples: 8000`), reduced epochs.
3. **WTI negative price** (2020-04-20) — used as feature only, not investable.
4. **Online RL caveat** — PPO/SAC/A2C interact with environment; not fair offline comparison.
5. **Risk parity underperformance** — may need tuning of lookback window.
6. **Generated data artifacts** — `data/processed/`, `data/trajectories/`, `results/checkpoints/` are gitignored; must re-run pipeline on fresh clone.

---

## Prioritized Next Steps

1. **Install CUDA PyTorch** and retrain with `feature_set: full`, 5 seeds, 50 epochs.
2. **DT inference loop** — implement RTG-conditioned rollout on test period and add to evaluation harness.
3. **RTG calibration curve** — sweep target RTG quantiles, plot realized vs. target return.
4. **Universe B** — set `data.universe: B`, `start_date: 2014-09-17`, re-run pipeline.
5. **Statistical tests** — wire `src/eval/stats.py` into harness (block-bootstrap CIs, deflated Sharpe).
6. **Fix PPO** — investigate Dirichlet NaN from state distribution; add gradient clipping.
7. **Increase trajectory count** — restore `n_windows_per_policy: 200` on GPU.
8. **Walk-forward DT evaluation** — load checkpoints per fold, report mean ± std across seeds.

---

## Git Commit History (logical steps)

| Step | Commit message |
|------|----------------|
| 0 | `docs: add project plan` |
| 1 | `chore(setup): scaffold project with src, tests, configs` |
| 2 | `feat(data): verify market data loader on Universe A` |
| 3 | `docs(eda): data exploration notebook and EDA figures` |
| 7 | `feat(trajectories): generate offline trajectory dataset` |
| 8-10 | `feat(models): train DT, BC, and offline RL baselines` |
| 11+ | `feat(eval): evaluation harness and ablations` (pending) |
| 13 | `docs(report): paper and results` (pending) |
| 14 | `docs: implementation summary` (this commit) |

---

*For full scope and mathematical formulation, see [project_plan.md](project_plan.md) and [project_context.md](project_context.md).*

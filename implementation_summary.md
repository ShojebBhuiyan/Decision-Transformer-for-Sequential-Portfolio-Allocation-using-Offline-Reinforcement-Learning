# Implementation Summary

**Project:** Decision Transformer for Sequential Portfolio Allocation  
**Date:** 2026-08-21  
**Status:** GPU-optimized pipeline ready for full-scale experiments

---

## What Was Built

A full offline RL research pipeline in `F:\Research\RL\` framing multi-asset portfolio allocation as sequence modeling with a Decision Transformer, trained on synthesized trajectories from the 30-year daily close dataset.

### Repository layout

| Path | Purpose |
|------|---------|
| [`src/data_loader.py`](src/data_loader.py) | CSV loading, NYSE calendar alignment, Universe A/B, synthetic bond & cash |
| [`src/features.py`](src/features.py) | State engineering with fingerprinted cache (`build_or_load_features`) |
| [`src/env.py`](src/env.py) | Portfolio environment with log reward & batched simplex projection |
| [`src/policies.py`](src/policies.py) | Classical + noisy behavior policies with precomputed schedules |
| [`src/trajectories.py`](src/trajectories.py) | Format v2 trajectories, vectorized rollouts, `GPUTrajectoryBuffer` |
| [`src/models/decision_transformer.py`](src/models/decision_transformer.py) | Causal GPT Decision Transformer (fixed attention masking) |
| [`src/models/bc.py`](src/models/bc.py) | Transformer BC + MLP-BC |
| [`src/models/offline_rl.py`](src/models/offline_rl.py) | TD3+BC, IQL, CQL |
| [`src/models/online_rl.py`](src/models/online_rl.py) | PPO, SAC, A2C (reference; uses interaction) |
| [`src/trainer.py`](src/trainer.py) | GPU-resident training loops with checkpointing |
| [`src/eval/`](src/eval/) | Backtest, metrics, stats, harness, ablations |
| [`configs/config.yaml`](configs/config.yaml) | Default hyperparameters (`batch_size: 256`) |
| [`main.py`](main.py) | Pipeline entrypoint |
| [`notebooks/`](notebooks/) | EDA and evaluation notebooks |
| [`paper/main.tex`](paper/main.tex) | LaTeX paper draft |
| [`tests/`](tests/) | 27 unit tests (all passing) |

---

## GPU Pipeline Optimization

### Trajectory format v2

| Field | Shape | Notes |
|-------|-------|-------|
| `states` | `(n_dates, state_dim)` | Global state matrix (shared across episodes) |
| `episode_starts` | `(n_episodes,)` | Index into `states` for each episode |
| `actions`, `rewards`, `rtg` | `(n_episodes, L, …)` | Per-episode sequences |
| `format_version` | `2` | v1 files must be regenerated |

### Data flow

1. **Policy schedules** precomputed once → cached at `data/processed/policy_weights.npz`
2. **Vectorized rollout** — episode rewards/RTG from precomputed weights (no per-day policy calls)
3. **Feature cache** — `states.npy` reused when fingerprint matches config
4. **GPUTrajectoryBuffer** — all tensors on device; batches built by index gather
5. **`last_state_only`** — TD3+BC, IQL, CQL, PPO, SAC, A2C skip K-context gather

### Attention correctness fix

**Before:** Right-aligned padding could mask all keys for a query row → softmax NaN → `nan_to_num` band-aid in trainer.  
**After:** Causal + padding mask combined; diagonal always attendable; `scaled_dot_product_attention`; no `nan_to_num` in trainer.

### AMP skipped on Pascal

GTX 1070 (compute 6.1) has no tensor cores and ~1/64 fp16 throughput vs fp32 on GP104. Mixed precision would likely slow training, so it is intentionally disabled.

### Revised runtime expectations

| Stage | Before (CPU-bound) | After (optimized) |
|-------|-------------------|-------------------|
| Trajectory generation | Tens of minutes | Few minutes |
| Training GPU utilization | 3–4% | GPU compute-bound |
| Full sweep (5 seeds × 8 models × 50 epochs) | N/A | Multi-hour to overnight |

Use `training.steps_per_epoch` to bound training runtime without code changes.

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
python main.py --stage trajectories # Synthesize offline trajectories (format v2, ~few min)
python main.py --stage train        # Train DT, BC, offline/online RL (GPU-bound)
python main.py --stage evaluate     # Backtest classical baselines
python main.py --stage ablations    # Transaction cost sweep + ablation manifest
```

Or run everything: `python main.py --stage all`

**Note:** After upgrading to format v2, delete old `data/trajectories/trajectories.npz` and re-run `--stage trajectories`.

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

Results tables: `results/tables/headline_test_metrics.csv`, `walkforward_metrics.csv`

---

## Known Limitations

1. **Close-only data** — no volume, no intraday features.
2. **WTI negative price** (2020-04-20) — used as feature only, not investable.
3. **Online RL caveat** — PPO/SAC/A2C interact with environment; not fair offline comparison.
4. **Risk parity underperformance** — may need tuning of lookback window.
5. **Noisy policy RNG** — vectorized Dirichlet (gamma trick) is statistically equivalent but not bit-identical to per-day `rng.dirichlet`.
6. **Generated data artifacts** — `data/processed/`, `data/trajectories/`, `results/checkpoints/` are gitignored; must re-run pipeline on fresh clone.

---

## Prioritized Next Steps

1. **Run full training sweep** — 5 seeds × 8 models × 50 epochs with format v2 trajectories.
2. **DT inference loop** — implement RTG-conditioned rollout on test period and add to evaluation harness.
3. **RTG calibration curve** — sweep target RTG quantiles, plot realized vs. target return.
4. **Universe B** — set `data.universe: B`, `start_date: 2014-09-17`, re-run pipeline.
5. **Statistical tests** — wire `src/eval/stats.py` into harness (block-bootstrap CIs, deflated Sharpe).
6. **Fix PPO** — investigate Dirichlet NaN from state distribution; add gradient clipping.
7. **Walk-forward DT evaluation** — load checkpoints per fold, report mean ± std across seeds.

---

## Git Commit History (GPU optimization)

| Step | Commit message |
|------|----------------|
| 1 | `perf(env): add batched simplex projection` |
| 2 | `perf(policies): precompute and cache behavior weight schedules` |
| 3 | `perf(trajectories): vectorize rollouts and add format v2` |
| 4 | `fix(model): correct attention padding mask producing NaN` |
| 5 | `perf(train): GPU-resident batch sampler and loop tuning` |
| 6 | `perf(features): fingerprinted feature cache` |
| 7 | `test: equivalence and regression coverage for optimizations` |
| 8 | `docs: record optimization architecture and revised runtimes` |

---

*For full scope and mathematical formulation, see [project_plan.md](project_plan.md) and [project_context.md](project_context.md).*

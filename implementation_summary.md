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
| [`scripts/verify_training.py`](scripts/verify_training.py) | Smoke-trains every model and reloads each checkpoint |
| [`tests/`](tests/) | 41 unit tests (all passing) |

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

### Measured runtime on a GTX 1070

Benchmarked at `d_model=192`, `n_layers=4`, `K=30` (90 tokens), 816,480 training samples:

| Batch size | ms/step | samples/s | Peak VRAM | Full epoch |
|-----------:|--------:|----------:|----------:|-----------:|
| 128 | 101 | 1,269 | 903 MB | 10.7 min |
| 256 | 196 | 1,306 | 1,600 MB | 10.4 min |
| 512 | 393 | 1,302 | 2,989 MB | 10.5 min |
| 1024 | 785 | 1,305 | 5,734 MB | 10.4 min |

Throughput plateaus at ~1,300 samples/s, so the GPU is genuinely compute-saturated;
larger batches buy nothing but memory. Batch 256 is the default.

Cost breakdown per step at batch 128: 103 ms model, 13 ms CPU gather
(1 ms when `gpu_resident_buffer: true`). Padding-masked attention costs about
10% over the fused causal path, so a fully-valid mask is dropped to reach it.

RL agents are far cheaper: 4–9 ms/step (0.2–0.5 min per full epoch).

**A full-pass sweep is not feasible on this hardware:** 5 seeds × 2 transformer
models × 50 epochs at 10.4 min/epoch is ~87 hours, plus ~52 hours of ablation
retraining. `training.steps_per_epoch: 200` keeps 50 epochs × 200 steps × 256 =
2.56M samples (~3 full passes) and brings the sweep into an overnight window.

| Knob | Default | Effect |
|------|---------|--------|
| `training.steps_per_epoch` | 200 | `null` = full pass (3189 steps @ batch 256) |
| `training.max_val_batches` | 40 | `null` = all 354 validation batches |
| `training.gpu_resident_buffer` | true | `false` = CPU storage + per-batch copy |

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

## RL correctness fixes

The RL baselines had defects that made their reported losses meaningless. All are
now covered by `tests/test_training_regressions.py`.

| Issue | Before | After |
|-------|--------|-------|
| Bellman target | `next_state = state`, so `Q(s) = r + γQ(s)` diverged | Real next state from `episode_starts`, with `dones` cutting the bootstrap at episode end |
| TD3+BC target critic | Deep-copied once, never updated | Polyak update, `tau = 0.005` |
| IQL value target | `V` fit against the live `Q`, chasing itself (loss → 10⁴) | `V` fit against a frozen target critic (loss ≈ 5e-4) |
| CQL penalty | `logsumexp` over the **batch**, so the loss was ~`2·log(batch)` | `logsumexp` over 10 sampled actions per state (CQL(H)) |
| CQL actor | Unbounded `-Q` maximization, drifting to -16 | BC anchor added, stays bounded |
| RL checkpoints | Saved only the algorithm **name string** | All `nn.Module` weights under `modules` |
| PPO / A2C | Crashed on pre-v2 batch shapes | Consume `last_state_only` batches; verified stable over 400+ steps |

## Known Limitations

1. **Close-only data** — no volume, no intraday features.
2. **WTI negative price** (2020-04-20) — used as feature only, not investable.
3. **Online RL caveat** — PPO/SAC/A2C are single-step adaptations over an offline buffer, not true environment interaction; treat them as reference points, not a fair online comparison.
4. **Risk parity underperformance** — may need tuning of lookback window.
5. **Noisy policy RNG** — vectorized Dirichlet (gamma trick) is statistically equivalent but not bit-identical to per-day `rng.dirichlet`.
6. **Evaluation covers classical baselines only** — `src/eval/harness.py` does not yet backtest trained DT/BC/RL checkpoints, so training metrics do not reach the results tables.
7. **Generated data artifacts** — `data/processed/`, `data/trajectories/`, `results/checkpoints/` are gitignored; must re-run pipeline on fresh clone.

---

## Prioritized Next Steps

1. **Run full training sweep** — `python main.py --stage train` (5 seeds × 8 models × 50 epochs at `steps_per_epoch: 200`). Verify the manifest ends with 40 entries; it is now written after every model, so an interrupted sweep keeps its progress.
2. **DT inference loop** — implement RTG-conditioned rollout on test period and add to evaluation harness. Until this lands, `results/tables/` reflects classical baselines only.
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

# Decision Transformer for Sequential Portfolio Allocation

**Project plan (in-repo reference)**  
Last updated: 2026-08-20

---

## 1. Overview

Frame multi-asset portfolio allocation as an offline sequence-modeling problem using a **Decision Transformer (DT)**. Train on synthesized trajectories from 30 years of daily close prices, conditioned on Return-to-Go (RTG). Benchmark against classical strategies, Behavior Cloning, offline RL (CQL, IQL, TD3+BC), and online RL (PPO, SAC, A2C).

**Hardware:** NVIDIA GTX 1070 (8 GB), Python 3.11, CUDA PyTorch  
**Stack:** PyTorch (custom in-repo implementations only; no FinRL / d3rlpy / stable-baselines3)

---

## 2a. GPU pipeline optimization (2026-08-21)

Three CPU bottlenecks were removed so training is GPU-bound rather than DataLoader-bound:

1. **Precomputed behavior-policy schedules** — deterministic policies (`MeanVariance`, `Momentum`, etc.) precompute `(T, N)` weight tables once; LedoitWolf stays per-date but gradient steps are batched. Cached at `data/processed/policy_weights.npz`.
2. **Trajectory format v2** — global `states (n_dates, state_dim)` + `episode_starts` replaces duplicated `(n_episodes, 252, state_dim)` (~9.4 GB → ~65 MB).
3. **GPU-resident batch sampler** — `GPUTrajectoryBuffer` uploads arrays once and gathers batches on-device; `last_state_only` mode for offline/online RL agents.

Additional fixes: attention padding mask NaN bug (diagonal always attendable); RTG stats masked by `lengths`; fingerprinted feature cache; `batch_size: 256`, `cudnn.benchmark=True`. **AMP deliberately skipped** on Pascal (GTX 1070, compute 6.1, no tensor cores).

Measured on a GTX 1070: ~1,300 samples/s regardless of batch size (GPU compute-saturated), so a full-pass 5-seed sweep would take ~87 h. `training.steps_per_epoch: 200` keeps ~3 full passes across 50 epochs and fits overnight. See [implementation_summary.md](implementation_summary.md) for the benchmark table and the RL correctness fixes.

**Re-run required:** `python main.py --stage trajectories` after pulling (format v2 breaks v1 files).

---

## 2. Confirmed scope

| Decision | Choice |
|----------|--------|
| Asset universe | Diversified multi-asset (~8–12 investable + cash) |
| Rebalancing | Daily |
| Cash asset | Yes (13-week T-Bill yield / 252, floored at 0) |
| Transaction cost | `mu = 0.001` (0.1% per unit turnover) |
| Baselines | Classical + BC + offline RL + online RL + DT ablations |
| Evaluation | Fixed chronological holdout + walk-forward folds |
| Deliverables | `src/` package, notebooks, `results/`, LaTeX paper, `implementation_summary.md` |

---

## 3. Dataset

**Location:** [`dataset/`](dataset/)

| File | Description |
|------|-------------|
| `30_yr_market_data.csv` | Daily closing prices, Date as index (9,229 rows, 1995–2025) |
| `30_yr_symbols_data.csv` | Symbol metadata (type, unit) |
| `30_yr_financial_events.csv` | Major financial events for regime annotation |

### Key findings

- **Union calendar:** Only 7,801 rows have S&P500; crypto adds weekends → reindex to NYSE trading days (S&P500 non-null).
- **Coverage varies:** Gold/Silver/Copper/WTI from 2000-08-30; Bitcoin from 2014-09-17; Ethereum from 2017-11.
- **WTI negative close:** -37.63 on 2020-04-20 → WTI is a **feature only**, not investable.
- **Negative T-Bill yield:** -0.105% on 2020-03-26 → cash daily return floored at 0.
- **Close-only:** No volume → indicators must be close-based only.
- **Log reward guard:** `ln(a^T y - mu * turnover)` needs epsilon floor on crash days.

### Loading pattern

```python
yf_market_data = pd.read_csv(path + '/30_yr_market_data.csv', index_col=0)
yf_market_data.set_index(pd.DatetimeIndex(yf_market_data.index), inplace=True)
```

---

## 4. Universe definition

### Universe A (primary) — from 2000-09-01, action dim 14

| Category | Assets |
|----------|--------|
| Equity | S&P 500 ETF, Fidelity Growth Fund, Apple, Microsoft, Amazon, Nvidia, JP Morgan Chase, Walmart, Fidelity Energy Portfolio |
| Metals | Gold, Silver, Copper |
| Bond | Synthetic 10Y Treasury total return (from `T-Note 10 Years` yields) |
| Cash | 13-week T-Bill yield / 252 |

### Universe B (robustness) — from 2014-09-17

Universe A + Bitcoin + Aluminum.

### State features (non-investable)

CBOE Volatility, US Dollar, WTI, Brent, Natural Gas, DAX/FTSE/Hang Seng/Nasdaq/NYSE, Treasury term spreads, event-window flags from financial events CSV.

---

## 5. Train / validation / test splits

| Split | Period |
|-------|--------|
| Train | 2000-09-01 → 2016-12-31 |
| Validation | 2017-01-01 → 2019-12-31 |
| Test | 2020-01-01 → 2025-12-30 |

**Walk-forward:** 5 expanding-window folds with 2-year test blocks (2014–15, 2016–17, 2018–19, 2020–21, 2022–23) + final 2024–25 block.

**Seeds:** 5 per learned model; report mean ± std.

---

## 6. MDP formulation

- **State** `S_t = (w'_{t-1}, X_t, I_t)`: prior weights, lookback return matrix, technical/macro indicators.
- **Action** `a_t ∈ Δ^N`: portfolio weights (softmax output).
- **Reward** `R_t = ln(a_t^T y_t - μ Σ|w_{i,t} - w'_{i,t-1}|)` with epsilon guard.
- **RTG** `R̂_t = Σ_{t'=t}^T R_{t'}` (γ = 1.0).

---

## 7. Trajectory synthesis

**Behavior policies:** MVO (Ledoit-Wolf), Min-Variance, Risk Parity, Momentum, Equal-Weight, Buy-and-Hold + Dirichlet-noise variants + random Dirichlet rollouts.

**Episodes:** T = 252 days, stride 21, training range only. ~3,000 trajectories → `data/trajectories/*.npz`.

---

## 8. Models

| Model | Config |
|-------|--------|
| Decision Transformer | K=30, d_model=192, 4 layers, 6 heads, RTG+S+A tokens, softmax head |
| Behavior Cloning | Same backbone without RTG token + MLP-BC |
| Offline RL | TD3+BC, IQL, CQL (simplex actor, twin critics) |
| Online RL | PPO, SAC, A2C (training-period env only; caveat documented) |

**Inference:** RTG grid from training-return quantiles → calibration curve.

---

## 9. Evaluation metrics

Cumulative return, CAGR, vol, Sharpe, Sortino, Calmar, MDD, turnover, cost drag, VaR/CVaR, alpha/beta vs SPY. Statistical tests: block-bootstrap CIs, Ledoit-Wolf Sharpe test, deflated Sharpe. Regime slices via financial events.

### Ablations

- Context length K ∈ {10, 20, 30, 50}
- RTG on/off (DT vs BC)
- Behavior-policy mixture composition
- Transaction cost μ ∈ {0, 0.0005, 0.001, 0.0025}
- State features: returns-only vs full
- Universe B crypto robustness

---

## 10. Directory layout

```text
portfolio_dt/
├── data/
│   ├── raw/                 # symlink/copy of dataset/
│   ├── processed/
│   └── trajectories/
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── env.py
│   ├── policies.py
│   ├── trajectories.py
│   ├── dataset.py
│   ├── trainer.py
│   ├── models/
│   │   ├── decision_transformer.py
│   │   ├── bc.py
│   │   ├── offline_rl.py
│   │   └── online_rl.py
│   └── eval/
│       ├── backtest.py
│       ├── metrics.py
│       └── stats.py
├── configs/
│   └── config.yaml
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_evaluation_plots.ipynb
├── tests/
├── results/
├── paper/
├── requirements.txt
├── main.py
├── project_context.md
├── project_plan.md          # this file
└── implementation_summary.md
```

---

## 11. Working process

1. Write this `project_plan.md` first → **first commit**.
2. **One commit per logical step** below; message format `type(scope): summary`.
3. Do not commit failing `pytest`.
4. Large artifacts (checkpoints, trajectory shards) gitignored; metrics tables and paper sources tracked.
5. Write `implementation_summary.md` as **final commit**.

---

## 12. Work log

| Step | Task | Commit | Status |
|------|------|--------|--------|
| 0 | `project_plan.md` | `docs: add project plan` (dbb7a5c) | done |
| 1 | Scaffold repo, requirements, CUDA smoke test | `chore(setup): scaffold project` (a447d47) | done |
| 2 | `src/data_loader.py` | `feat(data): market data loader` | done |
| 3 | EDA notebook | `docs(eda): data exploration notebook` (3e9f4b1) | done |
| 4 | `src/features.py` + tests | (in setup commit a447d47) | done |
| 5 | `src/env.py`, backtest + tests | (in setup commit a447d47) | done |
| 6 | `src/policies.py` | (in setup commit a447d47) | done |
| 7 | `src/trajectories.py` | `feat(trajectories): generate offline trajectory dataset` (e634abe) | done |
| 8 | Decision Transformer + trainer | `feat(models): train DT, BC, and offline RL` (f13878e) | done |
| 9 | BC + offline RL | (in f13878e) | done |
| 10 | Online RL | (in f13878e, PPO partial) | done |
| 11 | Metrics, stats, eval harness | `feat(eval): evaluation harness and ablations` (ae222ba) | done |
| 12 | Ablation runs | (in ae222ba) | done |
| 13 | Paper, plots notebook, README | `docs(report): paper and results` (ae222ba) | done |
| 14 | `implementation_summary.md` | `docs: implementation summary` (f099ff2) | done |
| 15 | Batched simplex projection | `perf(env): add batched simplex projection` | done |
| 16 | Precomputed policy schedules + cache | `perf(policies): precompute and cache behavior weight schedules` | done |
| 17 | Vectorized rollouts + format v2 | `perf(trajectories): vectorize rollouts and add format v2` | done |
| 18 | Attention padding NaN fix | `fix(model): correct attention padding mask producing NaN` | done |
| 19 | GPU batch sampler + training tuning | `perf(train): GPU-resident batch sampler and loop tuning` | done |
| 20 | Fingerprinted feature cache | `perf(features): fingerprinted feature cache` | done |
| 21 | Optimization regression tests | `test: equivalence and regression coverage for optimizations` | done |
| 22 | Optimization docs | `docs: record optimization architecture and revised runtimes` | done |
| 23 | Epoch sampling + checkpoint/manifest fixes | `fix(train): correct epoch sampling, checkpoints and sweep resilience` | done |
| 24 | RL Bellman/target/CQL correctness | `fix(models): use real transitions and target networks in RL baselines` | done |
| 25 | Training regression tests | `test: cover RL correctness and checkpoint round-trip` | done |

---

## 13. Correctness safeguards (tests)

- Simplex projection / weight validity
- Reward = log net portfolio growth (hand-computed cases)
- Zero-cost buy-and-hold = raw returns
- No-lookahead on every feature
- RTG recursion: `RTG_t = R_t + RTG_{t+1}`
- Split-boundary leakage checks

---

*See also: [project_context.md](project_context.md) for full mathematical formulation.*

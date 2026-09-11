# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research pipeline for a Decision Transformer applied to long-only multi-asset portfolio
allocation, trained offline on trajectories synthesized from a 30-year daily-close CSV dataset
(`dataset/`). The deliverable is a paper (`paper/main.tex`) plus metric tables in
`results/tables/`, not a production service.

Math/MDP spec: `project_context.md`. Scope, splits and work log: `project_plan.md`.
Current state, measured runtimes and known limitations: `implementation_summary.md`.
Keep those in sync when behavior changes — they are the project's record.

## Commands

Windows venv at `.venv` (Python 3.12). PowerShell: `.venv\Scripts\activate`.
Torch is installed separately from `requirements.txt` (CUDA 11.8 wheel index).

```powershell
pytest tests/ -q                                  # 58 tests
pytest tests/test_env.py -q                       # one file
pytest tests/test_env.py::test_log_reward_hand_computed -q   # one test
python scripts/cuda_smoke_test.py                 # confirm CUDA visible
python scripts/verify_training.py                 # smoke-train every model, reload each checkpoint
```

Pipeline stages (each is idempotent; `--stage all` runs them in order):

```powershell
python main.py --stage prepare        # CSV -> aligned prices -> data/processed/
python main.py --stage trajectories   # features + behavior-policy rollouts -> trajectories.npz
python main.py --stage train          # 5 seeds x 8 models, resumable, ~8h on a GTX 1070
python main.py --stage evaluate       # classical + learned backtests, stats, RTG calibration
python main.py --stage ablations      # K sweep (retrains), transaction cost, Universe B classical
```

Universe B writes to scoped subdirectories and never touches Universe A artifacts:

```powershell
python scripts/run_universe_b.py --stage train
```

Figures: `python scripts/run_eda.py`, `python scripts/run_eval_plots.py` (they read the CSVs in
`results/tables/`, so run evaluation first). `run_eval_plots.py --universe B` resolves both the
tables it reads and the figures it writes through the config, so it plots Universe B into
`results/figures/eval/universe_B/`; `run_eda.py` is still Universe A only. The evaluation stage
itself emits only `rtg_calibration.png` — every other figure comes from these scripts, run by hand.

Paper-specific artifacts (the term paper reads these, the pipeline stages do not produce them):

```powershell
python scripts/run_paper_ablations.py --part cost   # ablation_transaction_cost.csv (no training)
python scripts/run_paper_ablations.py --part conc   # portfolio_concentration.csv (no training)
python scripts/run_paper_ablations.py --part k      # retrains DT at K=10/20/50, ~1h on a GTX 1070
python scripts/run_paper_figures.py                 # -> results/figures/paper/
```

`--part k` reuses `dt_seed42.pt` for K=30 (identical config) and tags the others
`dt_K{K}_seed42.pt`, so it never clobbers the main sweep; re-running skips any K whose
checkpoint exists.

Papers: `cd paper; pdflatex main.tex` (research draft) and
`cd term_paper; pdflatex main.tex; bibtex main; pdflatex main.tex; pdflatex main.tex`
(course term paper, `\graphicspath` points at `../results/figures/`).

## Architecture

Stages hand off through files, not in-memory objects — any stage runs standalone as long as the
previous stage's artifacts exist.

```
dataset/*.csv
  -> data_loader.load_market_data       NYSE calendar align, limited ffill (5d),
                                        synthetic 10Y bond from yields, CASH from T-bill
  -> data/processed/                    prices, states.npy, policy_weights.npz (+ fingerprint metas)
  -> features.build_state_features      prev weights + L-day return matrix + indicators + macro + event flags
  -> trajectories.generate_trajectories 18 behavior policies x N windows -> trajectories.npz (format v2)
  -> trainer.train_all_models           -> results/checkpoints/*.pt + results/training_manifest.json
  -> eval/harness.run_evaluation        -> results/tables/*.csv, eval_summary.json
```

**Config is a thin dict wrapper, not a schema.** `src/config.py` `Config.get("a", "b", default=...)`
walks `configs/config.yaml`. Never build paths directly — call `cfg.processed_dir()`,
`cfg.trajectories_dir()`, `cfg.checkpoint_dir()`, `cfg.tables_dir()`, `cfg.run_dir()`,
`cfg.training_manifest_path()`. Those apply `cfg.artifact_scope`: Universe A keeps the legacy flat
paths, Universe B appends `universe_B/`. A new artifact location means a new method there, or
Universe B runs will clobber Universe A results.

**Universe** is set by `data.universe` (A = 14 assets from 2000-09-01; B = A + Bitcoin + Aluminum
from 2014-09-17). `data_loader.universe_b_config(cfg)` derives the B config with the start date
and `splits.train_start` shifted. Many `dataset/` columns are feature-only (`FEATURE_ONLY_COLS`)
and deliberately not investable — WTI has a negative print in 2020.

**Environment / reward.** `src/env.py` is plain NumPy, no gym. Reward is net log growth
`ln(aᵀy − μ·turnover)` floored at `reward_epsilon`. `PortfolioEnv.run_policy(policy_fn)` drives
everything downstream: backtests, trajectory rollouts and learned-model evaluation all go through
a callable `(t, state, env) -> weights`, which is why `src/policies.py` classes and the
`src/eval/model_policy.py` checkpoint wrappers are interchangeable.

**Trajectory format v2** (`trajectories.npz`) stores one global `states` matrix plus per-episode
`episode_starts`, so episodes index into shared states instead of duplicating them. Bumping
`FORMAT_VERSION` requires deleting the old npz and re-running `--stage trajectories`.

**`GPUTrajectoryBuffer`** (`src/trajectories.py`) replaces a DataLoader: it holds all tensors on
GPU (or pinned CPU when `training.gpu_resident_buffer: false`), keeps a flat
`(episode_idx, local_t)` index over valid transitions, and assembles batches by index gather.
`last_state_only=True` skips the K-step context gather for the non-transformer agents. Its
train/val split is over *transitions* with a per-run seed — not the date-based train/val/test
split used at evaluation time.

**Models.** `DecisionTransformer` interleaves `(RTG, state, action)` tokens (3 per timestep) and
reads predictions off the state-token positions; `TransformerBC` is the same model with RTG
zeroed. A fully-valid padding mask is dropped so attention reaches the fused causal kernel — don't
reintroduce an always-on mask. Offline (`TD3BC`, `IQL`, `CQL`) and online (`PPO`, `SAC`, `A2C`)
agents are plain classes with `train_step(batch)` / `get_action(states)`, not `nn.Module`s;
`trainer._agent_state_dict` snapshots every `nn.Module` attribute into the checkpoint's `modules`
dict, and `eval/model_policy.load_agent` restores them by attribute name.

**Sweep resumability.** `train_all_models` rewrites `training_manifest.json` after every model,
skips keys whose checkpoint already exists, and records `"failed: ..."` instead of raising so one
broken model doesn't kill an overnight run. `training.force_retrain: true` ignores existing
checkpoints. Evaluation consumes that manifest — only non-failed entries get backtested.

**`DTPolicy` RTG semantics** (`src/eval/model_policy.py`): the target RTG is a quantile of training
episode starting RTG (`evaluation.default_rtg_quantile`), normalized with the `rtg_mean`/`rtg_std`
in `trajectory_meta.json`, decremented by realized log reward each step, and **reset every
`env.episode_length` (252) days** so multi-year rollouts stay in-distribution. The current-step
action slot is zero-filled at inference because training stores unshifted actions and relies on
the causal mask to hide them.

**Caching by fingerprint.** `features.build_or_load_features` and `policies.precompute_all_schedules`
reuse `states.npy` / `policy_weights.npz` only when a hash of the relevant config keys matches
(`_feature_fingerprint`, `_policy_weights_fingerprint`). A new config key that changes feature or
schedule content must be added to the fingerprint, or stale caches get reused silently.

## Constraints that shaped the code

- Target hardware is a GTX 1070 (Pascal, no tensor cores). AMP is intentionally **not** used —
  fp16 is ~1/64 fp32 throughput there. Throughput plateaus near 1300 samples/s regardless of
  batch size, so larger batches buy only VRAM usage.
- A full pass is 3189 steps/epoch; the 5-seed sweep would be ~87h. `training.steps_per_epoch: 200`
  and `training.max_val_batches: 40` cap it to an overnight run (`null` restores full passes).
- PPO/SAC/A2C are single-step adaptations over the offline buffer, not real environment
  interaction — reference points, not a fair online comparison. Don't report them as one.

## Conventions

- One commit per logical step, `type(scope): summary` (e.g. `perf(trajectories): ...`).
  Don't commit with failing tests.
- `data/processed/`, `data/trajectories/`, `results/checkpoints/`, `results/runs/` and
  `results/figures/` are gitignored; `results/tables/*.csv`, the manifests and `paper/` are
  tracked. A fresh clone must re-run the pipeline to regenerate artifacts.
- Tests use `pytest.importorskip("torch")` so the non-GPU suite still runs without torch.
  Correctness tests target the safeguards in `project_plan.md` §13 (simplex validity,
  hand-computed rewards, no-lookahead, RTG recursion, split-boundary leakage) and the RL fixes
  tabulated in `implementation_summary.md` (real Bellman next-states, Polyak targets, CQL(H)
  penalty over sampled actions).

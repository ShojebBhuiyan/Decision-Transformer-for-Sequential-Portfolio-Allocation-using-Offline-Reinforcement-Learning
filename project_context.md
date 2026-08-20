# Project Context: Decision Transformer for Sequential Portfolio Allocation

## 1. Project Overview
* **Project Name:** Decision Transformer for Sequential Portfolio Allocation using Offline Reinforcement Learning
* **Domain:** Quantitative Finance & Offline Reinforcement Learning (Offline RL)
* **Objective:** Frame multi-asset portfolio allocation as an offline sequence-modeling problem using a Decision Transformer (DT) architecture. Instead of dynamic programming or active market exploration, the model learns continuous allocation strategies directly from static, historical market trajectories conditioned on desired target returns (Return-to-Go).

---

## 2. Core Problem & Research Motivation
* **Problem:** Online RL algorithms (e.g., SAC, PPO) require environment interactions, which are impractical in financial markets due to real-money loss, transaction fee accumulation, and the impossibility of recreating historical market regimes.
* **Offline RL Challenge:** Standard off-policy RL (Q-learning, DDPG) suffers from severe extrapolation errors and distribution shifts when trained on static datasets without online rollouts.
* **Solution:** Decision Transformer treats RL as an autoregressive sequence-modeling task. By tokenizing context sequences of Returns-to-Go, States, and Actions, DT learns to generate allocation actions conditioned on specified performance targets ($\hat{R}_1$).

---

## 3. Technology Stack & Dependencies
* **Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch / PyTorch Lightning
* **Sequence Modeling:** Hugging Face `transformers` (or custom PyTorch Causal GPT implementation)
* **Financial Data & Environment:** `yfinance`, `FinRL`, `pandas`, `numpy`
* **Evaluation & Backtesting:** `QuantStats`, `scikit-learn`, `matplotlib`, `seaborn`

---

## 4. Mathematical Formulation (MDP & Trajectories)

### Asset Scope
A portfolio of $N$ assets (e.g., AAPL, MSFT, GOOGL, AMZN, TSLA) over an episode of $T$ trading periods.

### Relative Price Return Vector
$$y_t = \left[ \frac{p_{1,t}}{p_{1,t-1}}, \frac{p_{2,t}}{p_{2,t-1}}, \dots, \frac{p_{N,t}}{p_{N,t-1}} \right]^T$$

### State Space ($S_t$)
Defined as $S_t = (w'_{t-1}, X_t, I_t)$, where:
* $w'_{t-1} \in \Delta^N$: Portfolio weights prior to rebalancing at step $t$.
* $X_t \in \mathbb{R}^{N \times L}$: Relative return matrix over lookback window $L$.
* $I_t \in \mathbb{R}^M$: Technical indicators (e.g., MACD, RSI, Volatility metrics).

### Action Space ($A_t$)
Target allocation vector $a_t = [w_{1,t}, \dots, w_{N,t}]^T \in \mathbb{R}^N$ constrained to the simplex $\Delta^N$:
$$\sum_{i=1}^N w_{i,t} = 1 \quad \text{and} \quad w_{i,t} \ge 0 \quad \forall i$$

### Reward Function ($R_t$)
Net logarithmic return incorporating proportional transaction fee rate $\mu$:
$$R_t = \ln \left( a_t^T y_t - \mu \sum_{i=1}^N \left\vert{} w_{i,t} - w'_{i,t-1} \right\vert{} \right)$$

### Return-to-Go ($\hat{R}_t$)
Cumulative target future return remaining in the episode from time step $t$ to horizon $T$:
$$\hat{R}_t = \sum_{t'=t}^{T} \gamma^{t'-t} R_{t'} \quad (\text{with discount factor } \gamma = 1.0)$$

---

## 5. Offline Trajectory Dataset Synthesis
To train the model offline, historical price data is used to generate a rich, multi-policy trajectory dataset $\mathcal{D} = \{\tau_i\}_{i=1}^M$:

$$\tau = \Big( \hat{R}_1, S_1, A_1, R_1, \hat{R}_2, S_2, A_2, R_2, \dots, \hat{R}_T, S_T, A_T, R_T \Big)$$

### Behavior Policy Mixture
1. **Mean-Variance Optimization (MVO):** Classical Markowitz portfolio optimization.
2. **Momentum Strategy:** Allocating capital based on trailing relative asset momentum.
3. **Equal-Weight ($1/N$):** Baseline passive allocation across assets.
4. **Noisy/Suboptimal Rollouts:** Adding Gaussian/Dirichlet noise to simulate diverse trading behaviors and test policy stitching capabilities.

---

## 6. Model Architecture & Training

* **Architecture:** Causal GPT-style Transformer.
* **Context Length ($K$):** Sliding context window length (e.g., $K = 20$ to $50$ days).
* **Token Embeddings:** Separate linear projection layers for Returns-to-Go ($\hat{R}_t$), States ($S_t$), and Actions ($A_t$) mapped to embedding dimension $d_{\text{model}}$ with added learned positional encodings.
* **Softmax Output Layer:** Predicts target portfolio weights $a_t \in \Delta^N$ using a Softmax activation to strictly enforce simplex constraints.
* **Loss Function:** Autoregressive Mean Squared Error (MSE) on continuous action allocation vectors:
$$\mathcal{L}_{\text{MSE}}(\theta) = \mathbb{E}_{\tau \sim \mathcal{D}} \left[ \sum_{t=1}^T \left\Vert{} a_t - \hat{f}_{\theta}(\hat{R}_t, S_t, \dots, S_{t-K+1}) \right\Vert{}^2 \right]$$

---

## 7. Baseline Benchmarks & Metrics

### Baselines
* **Buy-and-Hold:** Static equal investment across chosen assets.
* **Equal-Weight ($1/N$):** Daily/monthly rebalancing to uniform weights.
* **Behavior Cloning (BC):** Standard supervised sequence modeling predicting $A_t$ given $S_t$ without Return-to-Go conditioning.

### Performance Metrics
* **Cumulative Return (%)**
* **Annualized Return (%)**
* **Sharpe Ratio** (Risk-adjusted performance)
* **Maximum Drawdown (MDD %)** (Peak-to-trough decline)

---

## 8. Suggested Directory Structure

```text
portfolio_dt/
├── data/
│   ├── raw/                 # Raw OHLCV market data (yfinance)
│   ├── processed/           # Feature-engineered indicators & states
│   └── trajectories/        # Synthesized offline trajectories (R, S, A)
├── src/
│   ├── data_loader.py       # Market data fetcher & technical indicator calculator
│   ├── dataset.py           # Dataset synthesis & PyTorch Dataset class
│   ├── models/
│   │   ├── decision_tf.py   # Decision Transformer architecture
│   │   └── baselines.py     # Behavior Cloning & MVO implementations
│   ├── trainer.py           # Offline training loop & loss logging
│   └── evaluate.py          # Backtesting engine & QuantStats reporting
├── configs/
│   └── config.yaml          # Hyperparameters (K, d_model, learning_rate, assets)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_evaluation_plots.ipynb
├── requirements.txt
├── project_context.md
└── main.py                  # Entrypoint for pipeline execution
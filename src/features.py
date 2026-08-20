"""State feature engineering: returns, technical indicators, macro features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config
from src.data_loader import MarketDataBundle, compute_price_returns, get_split_mask


@dataclass
class FeatureBundle:
    """Engineered features for RL states."""

    states: np.ndarray  # (T, state_dim)
    dates: pd.DatetimeIndex
    feature_names: list[str]
    norm_stats: dict[str, np.ndarray]
    lookback: int
    n_assets: int


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return (macd_line - signal_line) / (close + 1e-8)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def compute_bollinger_pctb(close: pd.Series, window: int = 20) -> pd.Series:
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return (close - lower) / (upper - lower + 1e-8)


def compute_realized_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window).std() * np.sqrt(252)


def cross_sectional_momentum_rank(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rank assets by trailing cumulative return (0-1 scale)."""
    cum_ret = (1 + returns).rolling(window).apply(lambda x: np.prod(x) - 1, raw=True)
    return cum_ret.rank(axis=1, pct=True)


def build_event_flags(dates: pd.DatetimeIndex, events: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """Binary flags for financial event windows."""
    flags = pd.DataFrame(0.0, index=dates, columns=[f"event_{i}" for i in range(len(events))])
    for i, row in events.iterrows():
        event_date = row["DATE"]
        start = event_date - pd.Timedelta(days=window_days)
        end = event_date + pd.Timedelta(days=window_days)
        mask = (dates >= start) & (dates <= end)
        flags.loc[mask, f"event_{i}"] = 1.0
    return flags


def build_macro_features(feature_prices: pd.DataFrame) -> pd.DataFrame:
    """Macro / market indicator features."""
    macro = pd.DataFrame(index=feature_prices.index)

    if "CBOE Volitility" in feature_prices.columns:
        macro["vix"] = feature_prices["CBOE Volitility"]
    if "US Dollar" in feature_prices.columns:
        macro["usd"] = feature_prices["US Dollar"].pct_change()
    if "T-Note 10 Years" in feature_prices.columns and "T-Bill 13 Week" in feature_prices.columns:
        macro["term_spread_10y_3m"] = (
            feature_prices["T-Note 10 Years"] - feature_prices["T-Bill 13 Week"]
        )
    if "T-Bond 30 Years" in feature_prices.columns and "T-Note 5 Years" in feature_prices.columns:
        macro["term_spread_30y_5y"] = (
            feature_prices["T-Bond 30 Years"] - feature_prices["T-Note 5 Years"]
        )
    if "Crude Oil-WTI" in feature_prices.columns:
        wti = feature_prices["Crude Oil-WTI"].clip(lower=1e-8)
        macro["wti_return"] = wti.pct_change()
    if "Natural Gas" in feature_prices.columns:
        macro["ng_return"] = feature_prices["Natural Gas"].pct_change()

    return macro.fillna(0.0)


def build_state_features(
    bundle: MarketDataBundle,
    lookback: int = 20,
    include_indicators: bool = True,
    include_macro: bool = True,
    include_events: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Build per-timestep feature matrix (before flattening lookback)."""
    prices = bundle.investable_prices
    returns = compute_price_returns(prices)

    feature_dfs: list[pd.DataFrame] = []
    names: list[str] = []

    # Per-asset returns (latest day)
    for col in returns.columns:
        feature_dfs.append(returns[[col]].rename(columns={col: f"ret_{col}"}))
        names.append(f"ret_{col}")

    if include_indicators:
        for col in prices.columns:
            close = prices[col]
            feature_dfs.append(compute_macd(close).to_frame(f"macd_{col}"))
            feature_dfs.append(compute_rsi(close).to_frame(f"rsi_{col}"))
            feature_dfs.append(compute_bollinger_pctb(close).to_frame(f"bb_{col}"))
            ret = close.pct_change()
            feature_dfs.append(compute_realized_vol(ret, 20).to_frame(f"vol20_{col}"))
            feature_dfs.append(compute_realized_vol(ret, 60).to_frame(f"vol60_{col}"))
            names.extend([f"macd_{col}", f"rsi_{col}", f"bb_{col}", f"vol20_{col}", f"vol60_{col}"])

        # Cross-sectional momentum ranks
        for window in (60, 120):
            ranks = cross_sectional_momentum_rank(returns, window)
            ranks.columns = [f"mom{window}_{c}" for c in ranks.columns]
            feature_dfs.append(ranks)
            names.extend(list(ranks.columns))

    if include_macro:
        macro = build_macro_features(bundle.feature_prices)
        feature_dfs.append(macro)
        names.extend(list(macro.columns))

    if include_events:
        events = build_event_flags(prices.index, bundle.events)
        feature_dfs.append(events)
        names.extend(list(events.columns))

    combined = pd.concat(feature_dfs, axis=1)
    combined = combined.iloc[lookback:].fillna(0.0)
    return combined, names


def flatten_lookback_window(features: pd.DataFrame, lookback: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Create sliding window states: each row is flattened [t-L+1, ..., t] features."""
    values = features.values
    n_rows = len(features) - lookback + 1
    if n_rows <= 0:
        raise ValueError(f"Not enough rows ({len(features)}) for lookback {lookback}")

    state_dim = features.shape[1] * lookback
    states = np.zeros((n_rows, state_dim), dtype=np.float32)
    for i in range(n_rows):
        window = values[i : i + lookback]
        states[i] = window.flatten()

    dates = features.index[lookback - 1 :]
    return states, dates


def compute_norm_stats(states: np.ndarray, train_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Compute mean/std on training subset only."""
    train_states = states[train_mask]
    return {
        "mean": train_states.mean(axis=0).astype(np.float32),
        "std": (train_states.std(axis=0) + 1e-8).astype(np.float32),
    }


def normalize_states(states: np.ndarray, norm_stats: dict[str, np.ndarray]) -> np.ndarray:
    normalized = (states - norm_stats["mean"]) / norm_stats["std"]
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_features(
    bundle: MarketDataBundle,
    cfg: Config,
    include_indicators: bool | None = None,
    include_macro: bool = True,
    include_events: bool = True,
) -> FeatureBundle:
    """Full feature pipeline."""
    feature_set = cfg.get("data", "feature_set", default="full")
    if include_indicators is None:
        include_indicators = feature_set != "minimal"
    lookback = int(cfg.get("data", "lookback", default=20))
    n_assets = len(bundle.universe)

    raw_features, feature_names = build_state_features(
        bundle,
        lookback=lookback,
        include_indicators=include_indicators,
        include_macro=include_macro,
        include_events=include_events,
    )
    states, dates = flatten_lookback_window(raw_features, lookback)

    train_mask = get_split_mask(
        dates,
        cfg.get("splits", "train_start", default="2000-09-01"),
        cfg.get("splits", "train_end", default="2016-12-31"),
    )
    norm_stats = compute_norm_stats(states, train_mask)
    states_norm = normalize_states(states, norm_stats)

    return FeatureBundle(
        states=states_norm,
        dates=dates,
        feature_names=feature_names,
        norm_stats=norm_stats,
        lookback=lookback,
        n_assets=n_assets,
    )


def save_features(fb: FeatureBundle, cfg: Config) -> Path:
    root = cfg.project_root
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "states.npy", fb.states)
    fb.dates.to_series().to_frame("date").to_parquet(out_dir / "state_dates.parquet")
    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(fb.feature_names, f, indent=2)
    np.savez(
        out_dir / "norm_stats.npz",
        mean=fb.norm_stats["mean"],
        std=fb.norm_stats["std"],
    )
    meta: dict[str, Any] = {
        "lookback": fb.lookback,
        "n_assets": fb.n_assets,
        "state_dim": fb.states.shape[1],
    }
    with open(out_dir / "feature_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_dir


def load_features(cfg: Config) -> FeatureBundle:
    root = cfg.project_root
    out_dir = root / "data" / "processed"

    states = np.load(out_dir / "states.npy")
    dates = pd.read_parquet(out_dir / "state_dates.parquet")["date"]
    dates = pd.DatetimeIndex(dates)
    with open(out_dir / "feature_names.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    ns = np.load(out_dir / "norm_stats.npz")
    with open(out_dir / "feature_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    return FeatureBundle(
        states=states,
        dates=dates,
        feature_names=feature_names,
        norm_stats={"mean": ns["mean"], "std": ns["std"]},
        lookback=meta["lookback"],
        n_assets=meta["n_assets"],
    )

"""Market data loading, calendar alignment, and universe construction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import Config

# Column names in 30_yr_market_data.csv (human-readable)
COL_SP500 = "S&P500"
COL_SP500_ETF = "S&P 500 ETF"
COL_VIX = "CBOE Volitility"
COL_TBILL = "T-Bill 13 Week"
COL_TNOTE10 = "T-Note 10 Years"
COL_TNOTE5 = "T-Note 5 Years"
COL_TBOND30 = "T-Bond 30 Years"
COL_WTI = "Crude Oil-WTI"
COL_BRENT = "Crude Oil-Brent"
COL_NG = "Natural Gas"
COL_USD = "US Dollar"
COL_GOLD = "Gold"
COL_SILVER = "Silver"
COL_COPPER = "Copper"
COL_BITCOIN = "Bitcoin"
COL_ALUMINUM = "Aluminum"

UNIVERSE_A_ASSETS = [
    COL_SP500_ETF,
    "Fidelity Growth Fund",
    "Apple",
    "Microsoft",
    "Amazon",
    "Nvidia",
    "JP Morgan Chase",
    "Walmart",
    "Fidelity Energy Portfolio",
    COL_GOLD,
    COL_SILVER,
    COL_COPPER,
    "SYNTH_BOND_10Y",
    "CASH",
]

UNIVERSE_B_EXTRA = [COL_BITCOIN, COL_ALUMINUM]

FEATURE_ONLY_COLS = [
    COL_VIX,
    COL_USD,
    COL_WTI,
    COL_BRENT,
    COL_NG,
    "DAX Index",
    "FTSE 100",
    "Hang Seng Index",
    "Nasdaq",
    "NYSE Composite",
    COL_SP500,
    COL_TBILL,
    COL_TNOTE5,
    COL_TNOTE10,
    COL_TBOND30,
]


@dataclass
class MarketDataBundle:
    """Container for aligned market data."""

    prices: pd.DataFrame
    investable_prices: pd.DataFrame
    feature_prices: pd.DataFrame
    cash_daily_return: pd.Series
    events: pd.DataFrame
    symbols_meta: pd.DataFrame
    universe: list[str]
    coverage_mask: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


def _project_root(cfg: Config) -> Path:
    return cfg.project_root


def load_raw_market_csv(dataset_dir: Path) -> pd.DataFrame:
    """Load market data CSV with Date as datetime index."""
    path = dataset_dir / "30_yr_market_data.csv"
    df = pd.read_csv(path, index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()


def load_symbols_meta(dataset_dir: Path) -> pd.DataFrame:
    path = dataset_dir / "30_yr_symbols_data.csv"
    return pd.read_csv(path)


def load_financial_events(dataset_dir: Path) -> pd.DataFrame:
    path = dataset_dir / "30_yr_financial_events.csv"
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df


def align_to_nyse_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only NYSE trading days (rows where S&P500 is non-null)."""
    mask = df[COL_SP500].notna()
    return df.loc[mask].copy()


def limited_forward_fill(df: pd.DataFrame, max_fill: int = 5) -> pd.DataFrame:
    """Forward-fill missing values up to max_fill consecutive days."""
    return df.ffill(limit=max_fill)


def compute_cash_daily_return(tbill_yield: pd.Series) -> pd.Series:
    """Convert annualized T-Bill yield (%) to daily simple return, floored at 0."""
    daily = (tbill_yield / 100.0) / 252.0
    return daily.clip(lower=0.0)


def compute_synthetic_bond_prices(tnote10_yield: pd.Series, duration: float = 8.0) -> pd.Series:
    """
    Approximate 10Y Treasury note total-return index from yield changes.

    Uses modified duration approximation:
        r_t ≈ -D * Δy_t + y_{t-1}/252
    where y is yield in decimal.
    """
    y = tnote10_yield / 100.0
    dy = y.diff().fillna(0.0)
    coupon = y.shift(1).fillna(y.iloc[0]) / 252.0
    daily_ret = (-duration * dy + coupon).fillna(0.0)
    prices = (1.0 + daily_ret).cumprod() * 100.0
    prices.name = "SYNTH_BOND_10Y"
    return prices


def build_investable_prices(
    df: pd.DataFrame,
    universe: list[str],
    cash_return: pd.Series,
) -> pd.DataFrame:
    """Build price matrix for investable assets including synthetic bond and cash."""
    out = pd.DataFrame(index=df.index)
    for asset in universe:
        if asset == "CASH":
            # Cash index: cumulative product of (1 + r_cash)
            out[asset] = (1.0 + cash_return).cumprod()
        elif asset == "SYNTH_BOND_10Y":
            out[asset] = compute_synthetic_bond_prices(df[COL_TNOTE10])
        else:
            out[asset] = df[asset].astype(float)
    return out


def get_universe(universe_key: str) -> list[str]:
    if universe_key.upper() == "B":
        return UNIVERSE_A_ASSETS + UNIVERSE_B_EXTRA
    return list(UNIVERSE_A_ASSETS)


def get_universe_start_date(universe_key: str) -> str:
    if universe_key.upper() == "B":
        return "2014-09-17"
    return "2000-09-01"


def load_market_data(cfg: Config) -> MarketDataBundle:
    """Full data loading pipeline."""
    root = _project_root(cfg)
    dataset_dir = root / cfg.get("data", "dataset_dir", default="dataset")
    universe_key = cfg.get("data", "universe", default="A")
    start_date = cfg.get("data", "start_date", default=get_universe_start_date(universe_key))
    end_date = cfg.get("data", "end_date", default="2025-12-30")
    max_fill = int(cfg.get("data", "max_forward_fill", default=5))

    raw = load_raw_market_csv(dataset_dir)
    aligned = align_to_nyse_calendar(raw)
    aligned = limited_forward_fill(aligned, max_fill=max_fill)

    cash_return = compute_cash_daily_return(aligned[COL_TBILL].astype(float))
    universe = get_universe(universe_key)

    investable = build_investable_prices(aligned, universe, cash_return)

    # Drop rows with any NaN in investable universe
    valid_mask = investable.notna().all(axis=1)
    investable = investable.loc[valid_mask]
    aligned = aligned.loc[investable.index]
    cash_return = cash_return.loc[investable.index]

    # Date range filter
    investable = investable.loc[start_date:end_date]
    aligned = aligned.loc[start_date:end_date]
    cash_return = cash_return.loc[start_date:end_date]

    feature_cols = [c for c in FEATURE_ONLY_COLS if c in aligned.columns]
    feature_prices = aligned[feature_cols].copy()

    # Coverage mask: fraction of non-null per column in raw aligned data
    coverage_mask = aligned.notna().astype(float)

    events = load_financial_events(dataset_dir)
    symbols_meta = load_symbols_meta(dataset_dir)

    metadata = {
        "universe": universe_key,
        "n_assets": len(universe),
        "start_date": str(investable.index[0].date()),
        "end_date": str(investable.index[-1].date()),
        "n_trading_days": len(investable),
    }

    return MarketDataBundle(
        prices=aligned,
        investable_prices=investable,
        feature_prices=feature_prices,
        cash_daily_return=cash_return,
        events=events,
        symbols_meta=symbols_meta,
        universe=universe,
        coverage_mask=coverage_mask,
        metadata=metadata,
    )


def save_processed_data(bundle: MarketDataBundle, cfg: Config) -> Path:
    """Persist processed data to data/processed/."""
    root = _project_root(cfg)
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle.investable_prices.to_parquet(out_dir / "investable_prices.parquet")
    bundle.feature_prices.to_parquet(out_dir / "feature_prices.parquet")
    bundle.cash_daily_return.to_frame("cash_return").to_parquet(out_dir / "cash_return.parquet")

    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(bundle.metadata, f, indent=2)

    # Save universe list
    with open(out_dir / "universe.json", "w", encoding="utf-8") as f:
        json.dump(bundle.universe, f, indent=2)

    return out_dir


def load_processed_data(cfg: Config) -> MarketDataBundle:
    """Load previously processed data."""
    root = _project_root(cfg)
    out_dir = root / "data" / "processed"
    dataset_dir = root / cfg.get("data", "dataset_dir", default="dataset")

    investable = pd.read_parquet(out_dir / "investable_prices.parquet")
    features = pd.read_parquet(out_dir / "feature_prices.parquet")
    cash_return = pd.read_parquet(out_dir / "cash_return.parquet")["cash_return"]

    with open(out_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(out_dir / "universe.json", "r", encoding="utf-8") as f:
        universe = json.load(f)

    # Reconstruct minimal bundle
    events = load_financial_events(dataset_dir)
    symbols_meta = load_symbols_meta(dataset_dir)

    return MarketDataBundle(
        prices=investable,  # alias for compatibility
        investable_prices=investable,
        feature_prices=features,
        cash_daily_return=cash_return,
        events=events,
        symbols_meta=symbols_meta,
        universe=universe,
        coverage_mask=pd.DataFrame(),
        metadata=metadata,
    )


def compute_price_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Relative price return vector y_t = p_t / p_{t-1}."""
    returns = prices / prices.shift(1)
    # Guard against non-positive prices (e.g. WTI feature, not used for investable)
    returns = returns.clip(lower=1e-8)
    return returns.iloc[1:]


def get_split_mask(dates: pd.DatetimeIndex, start: str, end: str) -> np.ndarray:
    """Boolean mask for dates in [start, end]."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    return (dates >= start_dt) & (dates <= end_dt)

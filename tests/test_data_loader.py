"""Tests for data loading."""

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.data_loader import (
    align_to_nyse_calendar,
    compute_cash_daily_return,
    compute_synthetic_bond_prices,
    get_universe,
    load_market_data,
)


@pytest.fixture
def cfg():
    return Config.from_yaml("configs/config.yaml")


def test_universe_a_has_14_assets():
    universe = get_universe("A")
    assert len(universe) == 14
    assert "CASH" in universe
    assert "SYNTH_BOND_10Y" in universe


def test_universe_b_has_extra_assets():
    universe = get_universe("B")
    assert len(universe) == 16
    assert "Bitcoin" in universe
    assert "Aluminum" in universe


def test_align_to_nyse_calendar():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "S&P500": [100, np.nan, 102, 103, np.nan, 105, 106, 107, 108, 109],
        "Apple": range(10),
    }, index=idx)
    aligned = align_to_nyse_calendar(df)
    assert len(aligned) == 8  # only non-null S&P500 rows
    assert aligned["S&P500"].notna().all()


def test_cash_daily_return_floored():
    yields = pd.Series([-0.1, 0.0, 2.0, 5.0])
    daily = compute_cash_daily_return(yields)
    assert daily.iloc[0] == 0.0
    assert daily.iloc[1] == 0.0
    assert daily.iloc[2] > 0


def test_synthetic_bond_prices_positive():
    yields = pd.Series([3.0, 3.1, 2.9, 3.0, 3.2])
    prices = compute_synthetic_bond_prices(yields)
    assert (prices > 0).all()


def test_load_market_data(cfg):
    bundle = load_market_data(cfg)
    assert bundle.investable_prices.shape[1] == 14
    assert bundle.investable_prices.notna().all().all()
    assert bundle.metadata["n_trading_days"] > 1000

import numpy as np
import pandas as pd
import pytest

from megabot.data import synthetic_closes
from megabot.strategy import TrendConfig, target_weights, trend_signal, warmup_end


def _closes(values, columns=("A",)):
    index = pd.date_range("2020-01-01", periods=len(values), freq="D", tz="UTC")
    return pd.DataFrame(np.asarray(values, dtype=float).reshape(len(values), -1), index=index, columns=list(columns))


def test_signal_is_nan_until_longest_lookback_is_available():
    closes = _closes(np.arange(1, 31))
    signal = trend_signal(closes, (5, 10))
    assert signal["A"].iloc[:9].isna().all()
    assert signal["A"].iloc[9:].notna().all()


def test_uptrend_is_fully_long_and_downtrend_is_flat_or_short():
    up = _closes(np.arange(1, 61))
    down = _closes(np.arange(60, 0, -1))
    assert trend_signal(up, (5, 20)).iloc[-1, 0] == 1.0
    assert trend_signal(down, (5, 20), long_only=True).iloc[-1, 0] == 0.0
    assert trend_signal(down, (5, 20), long_only=False).iloc[-1, 0] == -1.0


def test_no_lookahead_future_prices_do_not_change_past_weights():
    closes = synthetic_closes(["A", "B", "C"], periods=500, seed=1)
    config = TrendConfig(lookbacks=(20, 50))
    cutoff = 300

    original = target_weights(closes, config)
    shocked = closes.copy()
    shocked.iloc[cutoff + 1 :] *= 10.0
    after_shock = target_weights(shocked, config)

    pd.testing.assert_frame_equal(original.iloc[: cutoff + 1], after_shock.iloc[: cutoff + 1])


def test_weights_respect_caps():
    closes = synthetic_closes(["A", "B", "C", "D"], periods=600, seed=2)
    config = TrendConfig(lookbacks=(20, 50), target_vol=5.0, max_weight=0.3, max_gross=1.0)
    weights = target_weights(closes, config)
    assert (weights.abs() <= 0.3 + 1e-12).all().all()
    assert (weights.abs().sum(axis=1) <= 1.0 + 1e-12).all()
    assert (weights >= 0).all().all()


def test_more_volatile_asset_gets_smaller_weight():
    rng = np.random.default_rng(0)
    n = 400
    trend = np.linspace(0, 1.0, n)
    calm = 100 * np.exp(trend + np.cumsum(rng.normal(0, 0.01, n)))
    wild = 100 * np.exp(trend + np.cumsum(rng.normal(0, 0.04, n)))
    closes = _closes(np.column_stack([calm, wild]), columns=("calm", "wild"))
    weights = target_weights(closes, TrendConfig(lookbacks=(20,), max_weight=1.0))
    both_long = (weights["calm"] > 0) & (weights["wild"] > 0)
    assert both_long.sum() > 50
    assert (weights.loc[both_long, "calm"] > weights.loc[both_long, "wild"]).all()


def test_warmup_end_raises_when_history_is_too_short():
    with pytest.raises(ValueError):
        warmup_end(_closes(np.arange(1, 11)), (20,))

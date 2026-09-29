import numpy as np
import pandas as pd
import pytest

from megabot.backtest import buy_and_hold, run_backtest
from megabot.metrics import cagr, compute_metrics, drawdown, sharpe


def _frame(data):
    index = pd.date_range("2021-01-01", periods=len(next(iter(data.values()))), freq="D", tz="UTC")
    return pd.DataFrame(data, index=index, dtype=float)


def test_buy_and_hold_without_costs_matches_price():
    closes = _frame({"A": [100, 110, 99, 120, 150]})
    result = buy_and_hold(closes["A"], fee=0.0, slippage=0.0)
    assert result.equity.iloc[-1] == pytest.approx(1.5)


def test_costs_are_charged_on_turnover():
    closes = _frame({"A": [100, 100, 100]})
    result = buy_and_hold(closes["A"], fee=0.001, slippage=0.0005)
    assert result.turnover.iloc[0] == pytest.approx(1.0)
    assert result.equity.iloc[-1] == pytest.approx(1.0 - 0.0015)


def test_target_on_bar_t_only_affects_return_of_bar_t_plus_1():
    closes = _frame({"A": [100, 200, 200, 400]})
    targets = _frame({"A": [0, 1, 0, 0]})
    result = run_backtest(closes, targets, fee=0.0, slippage=0.0)
    # Покупка на закрытии бара 1 (после роста до 200) не должна поймать этот рост.
    np.testing.assert_allclose(result.returns.to_numpy(), [0.0, 0.0, 0.0, 0.0])


def test_weights_drift_between_rebalances():
    closes = _frame({"A": [100, 200, 200], "B": [100, 100, 50]})
    targets = _frame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]})
    # Отклонение веса после первого бара 0.167 < порога 0.2, поэтому ребалансировки нет, веса только дрейфуют.
    result = run_backtest(closes, targets, fee=0.0, slippage=0.0, rebalance_threshold=0.2)
    assert result.equity.iloc[-1] == pytest.approx(0.5 * 2.0 + 0.5 * 0.5)
    assert result.weights.iloc[1]["A"] == pytest.approx(2 / 3)


def test_small_changes_are_skipped_but_full_exit_always_happens():
    closes = _frame({"A": [100, 100, 100, 100]})
    targets = _frame({"A": [0.5, 0.51, 0.005, 0.0]})
    result = run_backtest(closes, targets, fee=0.0, slippage=0.0, rebalance_threshold=0.02)
    np.testing.assert_allclose(result.weights["A"].to_numpy(), [0.5, 0.5, 0.005, 0.0])


def test_metrics_on_known_series():
    returns = pd.Series([0.1, -0.5, 1.0, 0.0], index=pd.date_range("2021-01-01", periods=4, freq="D", tz="UTC"))
    equity = (1 + returns).cumprod()
    np.testing.assert_allclose(drawdown(equity).to_numpy(), [0.0, -0.5, 0.0, 0.0])
    assert cagr(pd.Series([0.0] * 364 + [1.0]), 365) == pytest.approx(1.0)
    assert np.isnan(sharpe(pd.Series([0.01] * 10), 365))


def test_compute_metrics_runs_on_backtest_result():
    closes = _frame({"A": np.linspace(100, 200, 400)})
    result = buy_and_hold(closes["A"])
    metrics = compute_metrics(result, 365)
    assert metrics["total_return"] == pytest.approx(2.0 * (1 - 0.0015) - 1)
    assert metrics["max_drawdown"] == pytest.approx(-0.0015)
    assert metrics["avg_exposure"] == pytest.approx(1.0)

"""Метрики качества стратегии."""

from __future__ import annotations

import numpy as np
import pandas as pd

from megabot.backtest import BacktestResult


def drawdown(equity: pd.Series) -> pd.Series:
    """Просадка от предыдущего максимума капитала (0 или отрицательная).

    Начальный капитал 1.0 тоже считается максимумом: убыток на первом баре — уже просадка.
    """
    return equity / equity.cummax().clip(lower=1.0) - 1.0


def cagr(returns: pd.Series, periods_per_year: int) -> float:
    years = len(returns) / periods_per_year
    final = float((1.0 + returns).prod())
    if years <= 0 or final <= 0:
        return -1.0 if final <= 0 else float("nan")
    return final ** (1.0 / years) - 1.0


def sharpe(returns: pd.Series, periods_per_year: int) -> float:
    """Коэффициент Шарпа без безрисковой ставки."""
    std = returns.std(ddof=1)
    if np.isnan(std) or std < 1e-12:
        return float("nan")
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int) -> float:
    downside = np.sqrt((returns.clip(upper=0.0) ** 2).mean())
    if np.isnan(downside) or downside < 1e-12:
        return float("nan")
    return float(returns.mean() / downside * np.sqrt(periods_per_year))


def period_returns(returns: pd.Series, freq: str) -> pd.Series:
    """Доходности по календарным периодам: 'ME' — месяцы, 'YE' — годы."""
    return (1.0 + returns).resample(freq).prod() - 1.0


def compute_metrics(result: BacktestResult, periods_per_year: int) -> dict[str, float]:
    returns = result.returns
    years = len(returns) / periods_per_year
    equity = result.equity
    max_dd = float(drawdown(equity).min())
    annual = cagr(returns, periods_per_year)
    monthly = period_returns(returns, "ME")

    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": annual,
        "volatility": float(returns.std(ddof=1) * np.sqrt(periods_per_year)),
        "sharpe": sharpe(returns, periods_per_year),
        "sortino": sortino(returns, periods_per_year),
        "max_drawdown": max_dd,
        "calmar": annual / abs(max_dd) if max_dd < 0 else float("nan"),
        "positive_months": float((monthly > 0).mean()),
        "best_month": float(monthly.max()),
        "worst_month": float(monthly.min()),
        "avg_exposure": float(result.weights.abs().sum(axis=1).mean()),
        "turnover_per_year": float(result.turnover.sum() / years) if years > 0 else float("nan"),
        "costs_per_year": float(result.costs.sum() / years) if years > 0 else float("nan"),
    }

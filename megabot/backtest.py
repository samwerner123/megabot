"""Бэктест портфеля по целевым весам с учётом комиссий, проскальзывания и дрейфа весов."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.Series  # доходность портфеля за бар после издержек
    weights: pd.DataFrame  # веса после ребалансировки на закрытии бара
    turnover: pd.Series  # оборот за бар: сумма |сделок| в долях капитала
    costs: pd.Series  # издержки за бар в долях капитала

    @property
    def equity(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()


def run_backtest(
    closes: pd.DataFrame,
    targets: pd.DataFrame,
    fee: float = 0.001,
    slippage: float = 0.0005,
    rebalance_threshold: float = 0.02,
) -> BacktestResult:
    """Симуляция торговли по закрытиям.

    На каждом баре t:
    1. Портфель, собранный на закрытии t-1, получает доходность бара t.
       Веса дрейфуют вместе с ценами.
    2. На закрытии t сравниваем текущие веса с целевыми targets[t] (они
       посчитаны по данным до t включительно) и торгуем разницу.
       Мелкие сделки меньше rebalance_threshold пропускаем, чтобы не платить
       комиссии за шум; полный выход из позиции выполняется всегда.
    3. Комиссия и проскальзывание списываются с оборота.

    Так целевой вес бара t влияет только на доходность бара t+1:
    подглядывания в будущее нет.
    """
    targets = targets.reindex(index=closes.index, columns=closes.columns).fillna(0.0)
    rets = closes.pct_change(fill_method=None).fillna(0.0).to_numpy()
    tgt = targets.to_numpy()
    cost_rate = fee + slippage

    n_bars, n_assets = rets.shape
    w = np.zeros(n_assets)
    out_returns = np.zeros(n_bars)
    out_weights = np.zeros((n_bars, n_assets))
    out_turnover = np.zeros(n_bars)
    out_costs = np.zeros(n_bars)

    for t in range(n_bars):
        growth = 1.0 + float(w @ rets[t])
        if growth <= 0.0:
            # Капитал потерян полностью. Дальше торговать нечем.
            out_returns[t:] = 0.0
            out_returns[t] = -1.0
            break
        w = w * (1.0 + rets[t]) / growth

        trade = tgt[t] - w
        skip = np.abs(trade) < rebalance_threshold
        full_exit = (tgt[t] == 0.0) & (w != 0.0)
        trade[skip & ~full_exit] = 0.0

        turnover = float(np.abs(trade).sum())
        cost = turnover * cost_rate
        # Издержки уменьшают весь капитал пропорционально, веса остаются целевыми.
        w = w + trade

        out_returns[t] = growth * (1.0 - cost) - 1.0
        out_weights[t] = w
        out_turnover[t] = turnover
        out_costs[t] = cost

    index = closes.index
    return BacktestResult(
        returns=pd.Series(out_returns, index=index, name="returns"),
        weights=pd.DataFrame(out_weights, index=index, columns=closes.columns),
        turnover=pd.Series(out_turnover, index=index, name="turnover"),
        costs=pd.Series(out_costs, index=index, name="costs"),
    )


def buy_and_hold(close: pd.Series, fee: float = 0.001, slippage: float = 0.0005) -> BacktestResult:
    """Бенчмарк: купить один актив на весь капитал и держать."""
    closes = close.to_frame()
    targets = closes.notna().astype(float)
    return run_backtest(closes, targets, fee=fee, slippage=slippage)

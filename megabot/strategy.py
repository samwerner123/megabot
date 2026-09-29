"""Трендовая стратегия с размером позиции по волатильности.

Идея:
1. Сигнал: для нескольких окон (например, 20, 50, 100, 200 баров) смотрим,
   выше ли цена своей скользящей средней. Сигнал — доля окон «за»
   (от 0 до 1 для long-only, от -1 до 1 с шортами). Усреднение по окнам
   убирает зависимость от одного «удачного» параметра.
2. Размер позиции: чем выше волатильность монеты, тем меньше позиция, чтобы
   каждая монета вносила примерно одинаковый риск в портфель.
3. Ограничения: не больше max_weight на монету и не больше max_gross
   суммарно (1.0 = без плеча).

Все расчёты на баре t используют только данные до закрытия бара t включительно.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrendConfig:
    lookbacks: tuple[int, ...] = (20, 50, 100, 200)
    vol_span: int = 30
    target_vol: float = 0.40
    max_weight: float = 0.5
    max_gross: float = 1.0
    long_only: bool = True
    periods_per_year: int = 365


def trend_signal(closes: pd.DataFrame, lookbacks: tuple[int, ...], long_only: bool = True) -> pd.DataFrame:
    """Доля окон, в которых цена выше скользящей средней.

    NaN, пока истории не хватает на самое длинное окно.
    """
    if not lookbacks:
        raise ValueError("Нужно хотя бы одно окно")
    votes = []
    for lookback in lookbacks:
        sma = closes.rolling(lookback, min_periods=lookback).mean()
        above = (closes > sma).astype(float)
        vote = above if long_only else above * 2.0 - 1.0
        votes.append(vote.where(sma.notna() & closes.notna()))
    return sum(votes) / len(votes)


def realized_vol(closes: pd.DataFrame, span: int, periods_per_year: int) -> pd.DataFrame:
    """Годовая волатильность по экспоненциально взвешенному стандартному отклонению доходностей."""
    returns = closes.pct_change(fill_method=None)
    return returns.ewm(span=span, min_periods=span).std() * np.sqrt(periods_per_year)


def target_weights(closes: pd.DataFrame, config: TrendConfig) -> pd.DataFrame:
    """Целевые веса портфеля (доля капитала в каждой монете) на закрытии каждого бара."""
    signal = trend_signal(closes, config.lookbacks, config.long_only)
    vol = realized_vol(closes, config.vol_span, config.periods_per_year)

    valid = signal.notna() & vol.notna() & (vol > 0)
    n_valid = valid.sum(axis=1).replace(0, np.nan)
    risk_budget = config.target_vol / n_valid

    raw = (signal / vol).mul(risk_budget, axis=0)
    raw = raw.where(valid, 0.0).clip(-config.max_weight, config.max_weight)

    gross = raw.abs().sum(axis=1)
    scale = (config.max_gross / gross).clip(upper=1.0).fillna(1.0)
    return raw.mul(scale, axis=0)


def warmup_end(closes: pd.DataFrame, lookbacks: tuple[int, ...]) -> pd.Timestamp:
    """Первая дата, когда хотя бы у одной монеты хватает истории для сигнала."""
    ready = trend_signal(closes, lookbacks).notna().any(axis=1)
    if not ready.any():
        raise ValueError(f"Недостаточно истории: нужно больше {max(lookbacks)} баров")
    return ready.idxmax()

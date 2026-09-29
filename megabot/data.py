"""Загрузка исторических свечей с биржи (через ccxt) и локальный кэш в CSV."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")

TIMEFRAME_MS = {
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}

# Сколько баров в году. Крипта торгуется 24/7, поэтому 365 дней.
PERIODS_PER_YEAR = {
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
}

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def cache_path(exchange_id: str, symbol: str, timeframe: str, data_dir: Path = DATA_DIR) -> Path:
    safe = symbol.replace("/", "-").replace(":", "-")
    return data_dir / exchange_id / f"{safe}_{timeframe}.csv"


def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str = "1d", since: str = "2017-01-01") -> pd.DataFrame:
    """Скачивает всю историю свечей постранично.

    Последняя, ещё не закрытая свеча отбрасывается: её close ещё не финальный,
    и использование такой свечи в бэктесте было бы подглядыванием в будущее.
    """
    import ccxt  # импорт здесь, чтобы бэктест на кэше работал без ccxt

    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Неподдерживаемый таймфрейм {timeframe!r}, доступны: {sorted(TIMEFRAME_MS)}")

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    tf_ms = TIMEFRAME_MS[timeframe]
    since_ms = exchange.parse8601(f"{since}T00:00:00Z")
    now_ms = exchange.milliseconds()

    rows: list[list[float]] = []
    while since_ms < now_ms:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
        except ccxt.BaseError as exc:
            raise RuntimeError(
                f"Не удалось скачать {symbol} с {exchange_id}: {exc}\n"
                "Если биржа недоступна из вашей страны, попробуйте другую: --exchange bybit, okx или kraken."
            ) from exc
        if not batch:
            break
        rows.extend(batch)
        next_since = batch[-1][0] + tf_ms
        if next_since <= since_ms:
            break
        since_ms = next_since
        time.sleep(exchange.rateLimit / 1000)

    if not rows:
        raise RuntimeError(f"{exchange_id} не вернул данных по {symbol} {timeframe} с {since}")

    df = pd.DataFrame(rows, columns=["timestamp", *OHLCV_COLUMNS])
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df = df[df["timestamp"] + tf_ms <= now_ms]
    df.index = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
    return _normalize_index(df)


def save_ohlcv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=["date"])
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return _normalize_index(df)


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    """Одинаковый тип индекса у скачанных и закэшированных данных."""
    df.index = pd.DatetimeIndex(df.index).as_unit("us")
    df.index.name = "date"
    return df


def get_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str = "1d",
    since: str = "2017-01-01",
    refresh: bool = False,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Берёт свечи из кэша, при отсутствии (или refresh=True) скачивает с биржи."""
    path = cache_path(exchange_id, symbol, timeframe, data_dir)
    if path.exists() and not refresh:
        return load_ohlcv(path)
    df = fetch_ohlcv(exchange_id, symbol, timeframe, since)
    save_ohlcv(df, path)
    return df


def load_closes(
    exchange_id: str,
    symbols: list[str],
    timeframe: str = "1d",
    since: str = "2017-01-01",
    refresh: bool = False,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Таблица цен закрытия: строки — даты, столбцы — символы.

    До листинга монеты значения NaN, стратегия в это время её не торгует.
    """
    closes = {
        symbol: get_ohlcv(exchange_id, symbol, timeframe, since, refresh, data_dir)["close"]
        for symbol in symbols
    }
    df = pd.DataFrame(closes).sort_index()
    return df[df.index >= pd.Timestamp(since, tz="UTC")]


def synthetic_closes(
    symbols: list[str],
    start: str = "2018-01-01",
    periods: int = 365 * 7,
    seed: int = 7,
) -> pd.DataFrame:
    """Синтетические цены для демо и тестов без доступа к бирже.

    Случайное блуждание со сменой режимов (рост, падение, боковик) и общей
    рыночной компонентой, чтобы монеты были коррелированы, как в реальной крипте.
    Результат бэктеста на этих данных ничего не говорит о реальной доходности.
    """
    rng = np.random.default_rng(seed)
    n = len(symbols)
    daily_vol = 0.035

    regimes = np.array([0.0025, -0.0025, 0.0])
    regime_len = rng.integers(40, 160, size=periods)
    drift = np.empty(periods)
    i = 0
    while i < periods:
        length = int(regime_len[i])
        drift[i : i + length] = rng.choice(regimes, p=[0.4, 0.25, 0.35])
        i += length

    market = rng.normal(0.0, daily_vol * 0.8, size=periods)
    idio = rng.normal(0.0, daily_vol * 0.6, size=(periods, n))
    betas = rng.uniform(0.8, 1.4, size=n)
    log_rets = drift[:, None] * betas + market[:, None] * betas + idio

    prices = 100.0 * np.exp(np.cumsum(log_rets, axis=0))
    index = pd.date_range(start, periods=periods, freq="D", tz="UTC", name="date")
    return pd.DataFrame(prices, index=index, columns=symbols)

import ccxt
import pandas as pd
import pytest

from megabot.data import TIMEFRAME_MS, get_ohlcv, load_closes

DAY = TIMEFRAME_MS["1d"]
START = 1_577_836_800_000  # 2020-01-01T00:00:00Z


class FakeExchange:
    """Биржа-заглушка: 2500 дневных свечей, последняя ещё не закрыта."""

    rateLimit = 0
    n_candles = 2500

    def __init__(self, config):
        self.calls = 0
        self.now = START + (self.n_candles - 1) * DAY + DAY // 2

    def parse8601(self, text):
        return int(pd.Timestamp(text).value // 1_000_000)

    def milliseconds(self):
        return self.now

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        self.calls += 1
        first = max(0, (since - START + DAY - 1) // DAY)
        rows = []
        for i in range(first, min(first + limit, self.n_candles)):
            price = 100.0 + i
            rows.append([START + i * DAY, price, price + 1, price - 1, price, 10.0])
        return rows


@pytest.fixture
def fake_exchange(monkeypatch):
    monkeypatch.setattr(ccxt, "fakeex", FakeExchange, raising=False)


def test_fetch_paginates_and_drops_unfinished_candle(fake_exchange, tmp_path):
    df = get_ohlcv("fakeex", "BTC/USDT", "1d", since="2020-01-01", data_dir=tmp_path)
    assert len(df) == FakeExchange.n_candles - 1
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert df.index[0] == pd.Timestamp("2020-01-01", tz="UTC")
    assert df["close"].iloc[-1] == 100.0 + FakeExchange.n_candles - 2


def test_cache_roundtrip(fake_exchange, tmp_path):
    fetched = get_ohlcv("fakeex", "BTC/USDT", "1d", since="2020-01-01", data_dir=tmp_path)
    assert (tmp_path / "fakeex" / "BTC-USDT_1d.csv").exists()
    cached = get_ohlcv("fakeex", "BTC/USDT", "1d", since="2020-01-01", data_dir=tmp_path)
    pd.testing.assert_frame_equal(fetched, cached, check_freq=False)


def test_load_closes_builds_one_column_per_symbol(fake_exchange, tmp_path):
    closes = load_closes("fakeex", ["BTC/USDT", "ETH/USDT"], "1d", since="2021-01-01", data_dir=tmp_path)
    assert list(closes.columns) == ["BTC/USDT", "ETH/USDT"]
    assert closes.index[0] == pd.Timestamp("2021-01-01", tz="UTC")


class WindowedLateListingExchange(FakeExchange):
    """Как OKX: отдаёт свечи только внутри окна [since, since + 100 дней), монета листится на 250-й день."""

    listing = 250

    def fetch_ohlcv(self, symbol, timeframe, since, limit):
        first = max(self.listing, (since - START + DAY - 1) // DAY)
        last = min((since - START) // DAY + 100, self.n_candles)
        return [[START + i * DAY, 1.0, 1.0, 1.0, 1.0, 1.0] for i in range(first, last)]


def test_fetch_skips_empty_windows_before_listing(monkeypatch, tmp_path):
    monkeypatch.setattr(ccxt, "windowed", WindowedLateListingExchange, raising=False)
    df = get_ohlcv("windowed", "NEW/USDT", "1d", since="2020-01-01", data_dir=tmp_path)
    assert df.index[0] == pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=WindowedLateListingExchange.listing)
    assert len(df) == FakeExchange.n_candles - 1 - WindowedLateListingExchange.listing

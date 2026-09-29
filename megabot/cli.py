"""Командная строка: python -m megabot {fetch,backtest,sweep}."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from megabot.backtest import buy_and_hold, run_backtest
from megabot.data import PERIODS_PER_YEAR, get_ohlcv, load_closes, synthetic_closes
from megabot.metrics import compute_metrics
from megabot.report import metrics_table, save_chart, save_csv, yearly_table
from megabot.strategy import TrendConfig, target_weights, warmup_end

# Монеты, которые торгуются на Binance с 2017–2018 годов.
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", "LTC/USDT"]
DEMO_WARNING = "СИНТЕТИЧЕСКИЕ ДАННЫЕ: результат показывает работу кода, а не реальную доходность"


def _int_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(",") if part.strip())


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exchange", default="binance", help="биржа в ccxt: binance, bybit, okx, kraken... (по умолчанию binance)")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="торговые пары, первая — бенчмарк")
    parser.add_argument("--timeframe", default="1d", choices=sorted(PERIODS_PER_YEAR), help="таймфрейм свечей")
    parser.add_argument("--since", default="2017-01-01", help="начало истории, ГГГГ-ММ-ДД")
    parser.add_argument("--refresh", action="store_true", help="скачать заново, игнорируя кэш")


def _add_strategy_args(parser: argparse.ArgumentParser) -> None:
    defaults = TrendConfig()
    parser.add_argument("--demo", action="store_true", help="синтетические данные вместо биржи (проверка без интернета)")
    parser.add_argument("--lookbacks", type=_int_tuple, default=defaults.lookbacks, help="окна скользящих средних через запятую")
    parser.add_argument("--vol-span", type=int, default=defaults.vol_span, help="окно оценки волатильности, баров")
    parser.add_argument("--target-vol", type=float, default=defaults.target_vol, help="целевая годовая волатильность портфеля")
    parser.add_argument("--max-weight", type=float, default=defaults.max_weight, help="максимальная доля капитала в одной монете")
    parser.add_argument("--max-gross", type=float, default=defaults.max_gross, help="максимальная суммарная позиция, 1.0 = без плеча")
    parser.add_argument("--allow-short", action="store_true", help="разрешить шорты (нужны фьючерсы или маржа)")
    parser.add_argument("--fee", type=float, default=0.001, help="комиссия за сделку, доля (0.001 = 0.1%%)")
    parser.add_argument("--slippage", type=float, default=0.0005, help="проскальзывание, доля")
    parser.add_argument("--threshold", type=float, default=0.02, help="не торговать изменения веса меньше этого")
    parser.add_argument("--out", type=Path, default=Path("reports"), help="папка для отчёта")


def _load(args: argparse.Namespace) -> tuple[pd.DataFrame, int, str]:
    if args.demo:
        symbols = [f"SYN{i}" for i in range(1, 7)]
        return synthetic_closes(symbols), PERIODS_PER_YEAR["1d"], DEMO_WARNING
    closes = load_closes(args.exchange, args.symbols, args.timeframe, args.since, args.refresh)
    return closes, PERIODS_PER_YEAR[args.timeframe], f"{args.exchange}, {args.timeframe}"


def _config(args: argparse.Namespace, periods_per_year: int) -> TrendConfig:
    return TrendConfig(
        lookbacks=args.lookbacks,
        vol_span=args.vol_span,
        target_vol=args.target_vol,
        max_weight=args.max_weight,
        max_gross=args.max_gross,
        long_only=not args.allow_short,
        periods_per_year=periods_per_year,
    )


def cmd_fetch(args: argparse.Namespace) -> int:
    for symbol in args.symbols:
        df = get_ohlcv(args.exchange, symbol, args.timeframe, args.since, refresh=True)
        print(f"{symbol}: {len(df)} свечей, {df.index[0]:%Y-%m-%d} — {df.index[-1]:%Y-%m-%d}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    closes, periods_per_year, source = _load(args)
    config = _config(args, periods_per_year)
    costs = {"fee": args.fee, "slippage": args.slippage}

    targets = target_weights(closes, config)
    start = warmup_end(closes, config.lookbacks)
    strategy = run_backtest(closes.loc[start:], targets.loc[start:], rebalance_threshold=args.threshold, **costs)
    benchmark_symbol = closes.columns[0]
    benchmark = buy_and_hold(closes[benchmark_symbol].loc[start:], **costs)

    results = {"Тренд": strategy, f"Держать {benchmark_symbol}": benchmark}
    metrics = {name: compute_metrics(result, periods_per_year) for name, result in results.items()}

    print(f"Источник: {source}")
    print(f"Период: {start:%Y-%m-%d} — {closes.index[-1]:%Y-%m-%d}   Монеты: {', '.join(closes.columns)}")
    print(
        f"Окна: {config.lookbacks}  целевая волатильность: {config.target_vol:.0%}  "
        f"{'long/short' if not config.long_only else 'только long'}  "
        f"издержки: {args.fee + args.slippage:.2%} за сделку"
    )
    print()
    print(metrics_table(metrics))
    print()
    print("Доходность по годам")
    print(yearly_table(results))
    print()
    print(f"Целевые веса на {targets.index[-1]:%Y-%m-%d} (что бот держал бы сейчас)")
    latest = targets.iloc[-1]
    for symbol, weight in latest.items():
        print(f"  {symbol:<12} {weight:6.1%}")
    print(f"  {'кэш':<12} {1.0 - latest.sum():6.1%}")

    csv_path = save_csv(results, args.out)
    title = "Тренд против «купить и держать»" + (" — синтетические данные, не реальная доходность" if args.demo else "")
    chart_path = save_chart(results, args.out, title)
    print()
    print(f"Отчёт: {chart_path}, {csv_path}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Проверка устойчивости: как стратегия работает с разными окнами."""
    closes, periods_per_year, source = _load(args)
    base = _config(args, periods_per_year)

    variants = {f"окно {lb}": (lb,) for lb in args.sweep_lookbacks}
    variants[f"смесь {','.join(map(str, base.lookbacks))}"] = base.lookbacks
    longest = max(max(lbs) for lbs in variants.values())
    start = warmup_end(closes, (longest,))

    rows = {}
    for name, lookbacks in variants.items():
        targets = target_weights(closes, replace(base, lookbacks=lookbacks))
        result = run_backtest(
            closes.loc[start:], targets.loc[start:], fee=args.fee, slippage=args.slippage, rebalance_threshold=args.threshold
        )
        m = compute_metrics(result, periods_per_year)
        rows[name] = {
            "CAGR": f"{m['cagr']:.1%}",
            "Шарп": f"{m['sharpe']:.2f}",
            "Макс. просадка": f"{m['max_drawdown']:.1%}",
            "Оборот/год": f"{m['turnover_per_year']:.1f}",
        }

    print(f"Источник: {source}")
    print(f"Период: {start:%Y-%m-%d} — {closes.index[-1]:%Y-%m-%d}  (одинаковый для всех вариантов)")
    print()
    print(pd.DataFrame(rows).T.to_string())
    print()
    print("Хороший знак — когда соседние окна дают похожий результат.")
    print("Если прибыль есть только у одного окна, это, скорее всего, подгонка.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="megabot", description="Трендовый бот для крипты: данные, бэктест, отчёт")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="скачать историю свечей в кэш data/")
    _add_data_args(fetch)
    fetch.set_defaults(func=cmd_fetch)

    backtest = sub.add_parser("backtest", help="бэктест стратегии и сравнение с «купить и держать»")
    _add_data_args(backtest)
    _add_strategy_args(backtest)
    backtest.set_defaults(func=cmd_backtest)

    sweep = sub.add_parser("sweep", help="проверка устойчивости к выбору окна")
    _add_data_args(sweep)
    _add_strategy_args(sweep)
    sweep.add_argument("--sweep-lookbacks", type=_int_tuple, default=(10, 20, 50, 100, 150, 200, 300), help="окна для перебора")
    sweep.set_defaults(func=cmd_sweep)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

"""Текстовый отчёт, CSV и график капитала/просадки."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from megabot.backtest import BacktestResult
from megabot.metrics import drawdown, period_returns

METRIC_LABELS = {
    "total_return": ("Доходность за период", "pct"),
    "cagr": ("Среднегодовая доходность (CAGR)", "pct"),
    "volatility": ("Волатильность (годовая)", "pct"),
    "sharpe": ("Коэффициент Шарпа", "num"),
    "sortino": ("Коэффициент Сортино", "num"),
    "max_drawdown": ("Максимальная просадка", "pct"),
    "calmar": ("Calmar (CAGR / просадка)", "num"),
    "positive_months": ("Доля прибыльных месяцев", "pct"),
    "best_month": ("Лучший месяц", "pct"),
    "worst_month": ("Худший месяц", "pct"),
    "avg_exposure": ("Средняя загрузка капитала", "pct"),
    "turnover_per_year": ("Оборот в год (x капитала)", "num"),
    "costs_per_year": ("Издержки в год", "pct"),
}

# Цвета проверены валидатором палитры (различимы при дальтонизме, контраст к фону >= 3:1).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834"]


def _fmt(value: float, kind: str) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.1%}" if kind == "pct" else f"{value:.2f}"


def metrics_table(metrics: dict[str, dict[str, float]]) -> str:
    """Таблица метрик: строки — показатели, столбцы — стратегии."""
    names = list(metrics)
    rows = []
    for key, (label, kind) in METRIC_LABELS.items():
        rows.append([label, *(_fmt(metrics[name].get(key, float("nan")), kind) for name in names)])
    widths = [max(len(str(row[i])) for row in rows + [["", *names]]) for i in range(len(names) + 1)]
    header = "  ".join(["".ljust(widths[0]), *(name.rjust(widths[i + 1]) for i, name in enumerate(names))])
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append("  ".join([row[0].ljust(widths[0]), *(row[i + 1].rjust(widths[i + 1]) for i in range(len(names)))]))
    return "\n".join(lines)


def yearly_table(results: dict[str, BacktestResult]) -> str:
    yearly = pd.DataFrame({name: period_returns(r.returns, "YE") for name, r in results.items()})
    yearly.index = yearly.index.year
    yearly.index.name = "Год"
    return yearly.map(lambda v: f"{v:.1%}").to_string()


def save_csv(results: dict[str, BacktestResult], out_dir: Path) -> Path:
    """Дневной капитал и просадка по каждой стратегии — табличная версия графика."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {f"{name} {col}": series for name, r in results.items() for col, series in (("equity", r.equity), ("drawdown", drawdown(r.equity)))}
    )
    path = out_dir / "equity.csv"
    frame.to_csv(path, float_format="%.6f")
    return path


def save_chart(results: dict[str, BacktestResult], out_dir: Path, title: str) -> Path:
    """Два графика друг под другом: капитал (лог. шкала) и просадка."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator

    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = 150
    line_width = 2 * 72 / dpi  # 2 пикселя

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(10, 6.5), dpi=dpi, sharex=True, gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.18}
    )
    fig.patch.set_facecolor(SURFACE)

    for ax in (ax_eq, ax_dd):
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, linewidth=72 / dpi, linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(which="both", colors=INK_MUTED, labelsize=8, length=0)

    ends = []
    for (name, result), color in zip(results.items(), SERIES):
        equity = result.equity
        dd = drawdown(equity)
        label = f"{name}  ×{equity.iloc[-1]:.2f}"
        ax_eq.plot(equity.index, equity.values, color=color, linewidth=line_width, solid_capstyle="round", label=label)
        ax_dd.plot(dd.index, dd.values, color=color, linewidth=line_width, solid_capstyle="round")
        ends.append((label, equity.index[-1], equity.iloc[-1]))

    ax_eq.set_yscale("log", base=2)
    ax_eq.minorticks_off()
    ax_eq.yaxis.set_major_locator(LogLocator(base=2))

    # Подписи на концах линий, только если они не налезают друг на друга;
    # иначе итог остаётся в легенде.
    low, high = np.log2(ax_eq.get_ylim())
    positions = sorted((np.log2(value) - low) / (high - low) for _, _, value in ends)
    if all(b - a > 0.05 for a, b in zip(positions, positions[1:])):
        for label, x, value in ends:
            ax_eq.annotate(label, xy=(x, value), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK_SECONDARY)
    ax_eq.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"×{v:g}"))
    ax_dd.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax_dd.set_ylim(top=0.0)

    ax_eq.set_title(title, loc="left", fontsize=11, color=INK_PRIMARY, pad=18)
    ax_eq.text(0.0, 1.02, "Рост капитала (лог. шкала), ×1 = начальный капитал", transform=ax_eq.transAxes, fontsize=8, color=INK_SECONDARY)
    ax_dd.text(0.0, 1.08, "Просадка от максимума", transform=ax_dd.transAxes, fontsize=8, color=INK_SECONDARY)

    legend = ax_eq.legend(loc="upper left", frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.subplots_adjust(left=0.07, right=0.84, top=0.9, bottom=0.07)
    path = out_dir / "equity.png"
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path

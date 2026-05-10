from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ai.chart_context import calculate_rsi
from core.config import resolve_path


def render_candlestick_snapshot(
    df: pd.DataFrame,
    output_path: str | Path,
    symbol: str = "",
    timeframe: str = "",
    lookback: int = 200,
    show_rsi: bool = True,
) -> Path:
    if df.empty:
        raise ValueError("Cannot render an empty chart.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    frame = df.tail(lookback).copy()
    output = resolve_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    raw_time = frame["time"] if "time" in frame else pd.Series(frame.index, index=frame.index)
    time_values = pd.to_datetime(raw_time, utc=True, errors="coerce")
    if time_values.isna().any():
        x = np.arange(len(frame), dtype=float)
        use_dates = False
    else:
        x = mdates.date2num(time_values.dt.tz_convert(None))
        use_dates = True

    opens = frame["open"].astype(float).to_numpy()
    highs = frame["high"].astype(float).to_numpy()
    lows = frame["low"].astype(float).to_numpy()
    closes = frame["close"].astype(float).to_numpy()
    width = (np.median(np.diff(x)) * 0.65) if len(x) > 1 else 0.02
    if not np.isfinite(width) or width <= 0:
        width = 0.02

    if show_rsi:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(14, 8),
            dpi=140,
            sharex=True,
            gridspec_kw={"height_ratios": [3.5, 1]},
        )
        price_ax, rsi_ax = axes
    else:
        fig, price_ax = plt.subplots(1, 1, figsize=(14, 6), dpi=140)
        rsi_ax = None
    fig.patch.set_facecolor("#111827")
    for axis in ([price_ax, rsi_ax] if rsi_ax is not None else [price_ax]):
        axis.set_facecolor("#111827")
        axis.grid(True, color="#374151", linewidth=0.5, alpha=0.55)
        axis.tick_params(colors="#d1d5db", labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#374151")

    for pos, open_, high, low, close in zip(x, opens, highs, lows, closes):
        color = "#00b8a9" if close >= open_ else "#ff304f"
        price_ax.vlines(pos, low, high, color=color, linewidth=0.8)
        lower = min(open_, close)
        height = max(abs(close - open_), 1e-9)
        price_ax.add_patch(
            plt.Rectangle(
                (pos - width / 2, lower),
                width,
                height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.6,
            )
        )

    if "atr" in frame:
        clean_atr = frame["atr"].replace([np.inf, -np.inf], np.nan).dropna()
        latest_atr = float(clean_atr.iloc[-1]) if not clean_atr.empty else 0.0
        price_ax.axhspan(closes[-1] - latest_atr, closes[-1] + latest_atr, color="#2563eb", alpha=0.08)

    price_ax.axhline(closes[-1], color="#22d3ee", linestyle="--", linewidth=0.8, alpha=0.9)
    title = " ".join(part for part in (symbol, timeframe, "hybrid snapshot") if part)
    price_ax.set_title(title or "hybrid snapshot", color="#f9fafb", loc="left", fontsize=12)
    price_ax.yaxis.tick_right()

    if rsi_ax is not None:
        rsi = calculate_rsi(frame["close"].astype(float))
        rsi_ax.plot(x, rsi, color="#a78bfa", linewidth=1.0)
        rsi_ax.axhline(70, color="#9ca3af", linestyle="--", linewidth=0.7)
        rsi_ax.axhline(50, color="#6b7280", linestyle=":", linewidth=0.7)
        rsi_ax.axhline(30, color="#9ca3af", linestyle="--", linewidth=0.7)
        rsi_ax.set_ylim(0, 100)
        rsi_ax.yaxis.tick_right()

    if use_dates:
        date_axis = rsi_ax if rsi_ax is not None else price_ax
        date_axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        fig.autofmt_xdate(rotation=0)

    fig.tight_layout()
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output

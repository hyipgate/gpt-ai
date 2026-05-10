from __future__ import annotations

import numpy as np
import pandas as pd


def _directional_prices(frame: pd.DataFrame, direction: int) -> pd.DataFrame:
    prices = frame[["open", "high", "low", "close"]].astype(float).copy()
    if direction < 0:
        inverted = -prices
        prices["open"] = inverted["open"]
        prices["close"] = inverted["close"]
        prices["high"] = inverted["low"]
        prices["low"] = inverted["high"]
    return prices


def chart_tensor(
    df: pd.DataFrame,
    end_pos: int,
    direction: int,
    lookback: int = 96,
    height: int = 48,
    width: int = 48,
) -> np.ndarray:
    """Encode recent candles as a small local image tensor.

    Sell setups are price-inverted so the model learns "with-trade" chart
    quality instead of memorizing buy/sell direction separately.
    """
    start = max(0, end_pos - lookback + 1)
    frame = df.iloc[start : end_pos + 1].copy()
    if frame.empty:
        return np.zeros((height, width), dtype=np.float32)

    prices = _directional_prices(frame, direction)
    low = float(prices["low"].min())
    high = float(prices["high"].max())
    price_range = max(high - low, 1e-12)

    image = np.zeros((height, width), dtype=np.float32)
    count = len(prices)
    x_positions = np.linspace(0, width - 1, count).round().astype(int)

    def y_for(price: float) -> int:
        scaled = (price - low) / price_range
        return int(np.clip(round((height - 1) * (1 - scaled)), 0, height - 1))

    for x, row in zip(x_positions, prices.itertuples(index=False)):
        open_y = y_for(float(row.open))
        high_y = y_for(float(row.high))
        low_y = y_for(float(row.low))
        close_y = y_for(float(row.close))
        top_wick, bottom_wick = sorted((high_y, low_y))
        image[top_wick : bottom_wick + 1, x] = 0.45
        top_body, bottom_body = sorted((open_y, close_y))
        body_value = 1.0 if close_y <= open_y else 0.7
        left = max(0, x - 1)
        right = min(width, x + 2)
        image[top_body : bottom_body + 1, left:right] = body_value

    return image


def chart_feature_row(
    df: pd.DataFrame,
    end_pos: int,
    direction: int,
    lookback: int = 96,
    height: int = 48,
    width: int = 48,
) -> np.ndarray:
    image = chart_tensor(df, end_pos, direction, lookback=lookback, height=height, width=width)
    return image.reshape(-1)


def chart_feature_frame(
    df: pd.DataFrame,
    indices: pd.Index,
    directions: pd.Series,
    lookback: int = 96,
    height: int = 48,
    width: int = 48,
) -> pd.DataFrame:
    positions = {idx: pos for pos, idx in enumerate(df.index)}
    rows = []
    valid_index = []
    for idx in indices:
        if idx not in positions:
            continue
        direction = int(directions.loc[idx])
        if direction == 0:
            continue
        rows.append(chart_feature_row(df, positions[idx], direction, lookback=lookback, height=height, width=width))
        valid_index.append(idx)
    columns = [f"px_{i}" for i in range(height * width)]
    return pd.DataFrame(rows, index=valid_index, columns=columns, dtype=np.float32)

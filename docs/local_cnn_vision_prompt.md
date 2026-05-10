# Local CNN Vision Training Prompt

Goal: build a fully local chart-vision model that learns from chart screenshots and user corrections without any external API.

The model should behave like a visual trading assistant:

- Input: a chart image, symbol, timeframe, intended direction, and optional human note.
- Target: whether the setup was visually good or bad.
- Output: probability that the visual setup is acceptable, pass/fail at threshold, and metrics from held-out validation.

Teaching format:

```text
This chart is a BUY/SELL idea.
Label: good or bad.
Reason: short human note explaining what happened or why the setup should/should not be trusted.
```

Training principles:

- Good examples should include clean structure, clear continuation or reversal context, acceptable RSI/momentum, and obvious invalidation.
- Bad examples should include chop, late entries, buys into resistance, sells into support, weak momentum, or setups that looked tempting but failed.
- The model must learn from both wins and human corrections. Human labels override automatic trade outcomes when they conflict.
- Keep symbol/timeframe context because visual patterns can differ across instruments and candle durations.
- Never call a remote model or API. Training and inference must run locally.

Operational rule:

Use the CNN as a filter, not as the only decision maker. A trade should still pass structure, risk, spread, and multi-timeframe checks.

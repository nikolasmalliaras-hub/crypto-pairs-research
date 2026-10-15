# Crypto pairs trading research: cointegration, tested until it broke

**Question:** Can a retail trader make money trading cointegrated crypto pairs (long one coin, short the other) after real costs?

**Answer after ~10 tests on up to 7 years of data:** No. The pair selection finds a real but tiny statistical effect, and trading costs eat all of it. The effect has also faded since 2024.

This repo documents the whole process, including the dead ends, because a negative result tested properly is still a result.

## Method

1. Rolling cointegration (Engle-Granger) on a training window, then trading on the next unseen window
2. Entry on spread z-score, exit on total PnL target or stop
3. Rules: each coin in at most one pair at a time; fees, slippage and funding included
4. **Pass/fail criteria written before each run:** net t-stat clearly above 2, positive in both halves of the sample, better than a random placebo, and not dependent on a few trades

## Results

| Test | Data | Result |
|---|---|---|
| Intraday, 1h candles | Hyperliquid, 36 coins, ~200 days | All 12 parameter sets negative net |
| Daily candles | Hyperliquid, 29 coins, ~2 years | All 16 positive, **but** top 5 trades = 96% of profit |
| Same idea, longer history | Binance, 39 coins, 5 years | 15/16 negative. The Hyperliquid result was noise from a short sample |
| Random correlated pairs (placebo) | Binance, 5 years | Clearly worse than real selection, so selection *does* add something |
| Strict p-value filter | Binance, 5 years | Positive 2022-2024, negative 2024-2026: edge decayed |
| Out-of-sample (2019-2026) | Binance, 7 years | All negative |
| Hybrid scalping, hourly | Binance, 4 years, 1,000-2,000 trades per set | Gross edge ≈ +0.2% per trade = round-trip cost. Net ≈ 0 |
| Split by BTC volatility regime | Binance, 4 + 7 years | Mid-volatility looked good in-sample, **failed** out-of-sample |

<!-- Add 1-2 charts here: equity curves, real vs placebo -->

## What I learned

- A short sample with a few big winners can look like an edge. Longer history and placebo tests are what kill false positives.
- Setting the criteria **before** seeing results is the most important rule in the project.
- Statistical significance before costs is not the same as profitability after costs.

## Tech

Python, pandas, NumPy, statsmodels, Binance and Hyperliquid public APIs.

## Files

| File | What it does |
|---|---|
| `hl_hourly_v6.py` | Hybrid pairs test on multi-year hourly data: cointegration gate on 4h data, hourly z-score entries, target/stop grid, t-stats, bootstrap, half and yearly splits, random-pair placebo |
| `hl_regime_v8.py` | Same engine, with every trade tagged by the BTC volatility regime at entry (percentile vs. past values only, no look-ahead) and checked against criteria set in advance |

## Run it

```bash
pip install -r requirements.txt
python hl_hourly_v6.py --synthetic                                  # quick check on fake data
python hl_hourly_v6.py --years 4 --coins 30                         # real test
python hl_hourly_v6.py --years 4 --coins 30 --placebo --seed 1      # random-pair control
python hl_regime_v8.py --years 4 --coins 30                         # split by volatility regime
```

All data comes from Binance public market data, so no account or API key is needed. The first run downloads the data into a local cache folder.

**Known limits:** only coins that still trade today are included (survivorship bias), stops are checked on hourly closes, and funding is only modelled as a flat daily drag.

---
*Code written with AI assistance. Research design, testing criteria and conclusions are my own.*

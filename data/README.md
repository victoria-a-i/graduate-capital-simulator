# Data provenance

| File | What | Source | Retrieved |
|---|---|---|---|
| `sp500_shiller_monthly.csv` | S&P Composite monthly: price (monthly **average** of daily closes), annualised dividend, earnings, CPI, 1871-01 → 2026-09. Dividends stop at 2023-06; later rows are price-only from FRED. | Robert Shiller, Yale (http://www.econ.yale.edu/~shiller/data.htm), tidied by https://github.com/datasets/s-and-p-500 (`data/data.csv`, PDDL) | 2026-11-11 |
| `sp500tr_yahoo_monthly.csv` | ^SP500TR (S&P 500 Total Return index) monthly close, 1988-01 → 2024-12; Yahoo's monthly dates label each month and the Close is its month-end value | Downloaded directly from Yahoo Finance with yfinance by Victoria; the CSV retains the loader's expected layout, with unused OHLCV fields blank | 2026-11-16 |

## Why two sources
- **Shiller**: long history (1871–2023) with dividends → total returns. But month-average prices
  smooth returns: σ is understated by ≈√(2/3) and lag-1 autocorrelation is ≈ +0.25 (Working 1960).
  `scripts/real_data_test.py` corrects σ using non-overlapping annual returns.
- **Yahoo ^SP500TR**: true month-end total-return index, no smoothing, but only ~37 years.

## Verification
The Yahoo file was downloaded directly by Victoria, replacing the earlier third-party copy. Its
calendar-year total returns match the published figures to two decimal places for 2008
(−37.00%), 2022 (−18.11%), 2023 (+26.29%), and 2024 (+25.02%)
(`tests/test_historical.py`).

## Known limitations
- Pre-1957 "S&P 500" values are Cowles/S&P 90 reconstructions; pre-1926 dividends are interpolated from annual data.
- US-only. The US was among the best-performing equity markets of the 20th century
  (Jorion & Goetzmann 1999; Dimson, Marsh & Staunton 2002), so US history likely overstates what a
  global or forward-looking investor should expect.
- Nominal returns (consistent with nominal loan APRs). No taxes or fund fees.

# Step-by-step guide for a coding agent — weekly → multi-week stock prediction (with free data examples)

Below is a concise, ordered plan your coding agent can follow to produce a working MVP (data → features → model → backtest). Each step lists exact deliverables, example commands/snippets and where to obtain free data. I remain formal and concise.

---

## Project layout (create first)

```
repo/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ external/
├─ src/
│  ├─ ingestion/
│  ├─ preprocess/
│  ├─ features/
│  ├─ models/
│  ├─ backtest/
│  └─ utils/
├─ notebooks/
├─ tests/
├─ requirements.txt
└─ README.md
```

---

## Step 0 — Environment & dependencies

* Create virtual environment and minimal deps:

  * `python -m venv .venv && . .venv/bin/activate`
  * `pip install pandas numpy scipy pyarrow scikit-learn torch transformers sentence-transformers yfinance alpha_vantage fredapi pytrends requests beautifulsoup4 tqdm`
* Save packages to `requirements.txt`.

---

## Step 1 — Data acquisition (scripts in `src/ingestion/`)

Implement small scripts that each produce a cached file in `data/raw/` (CSV or Parquet). Provide a CLI or simple function interface.

### 1.1 Equity prices (daily → aggregate weekly)

* Preferred free sources to implement connectors for (choose one or implement multiple and reconcile):

  * **yfinance** (unofficial Python library for Yahoo Finance historical prices). Example: `yfinance.download(...)`. ([GitHub][1])
  * **Alpha Vantage** (free API key, limited calls; good for programmatic access). ([Alpha Vantage][2])
  * **Stooq** (free downloadable historical data; good fallback for many exchanges). ([Stooq][3])

Example (yfinance):

```python
import yfinance as yf
df = yf.download(['AAPL','MSFT'], start='2010-01-01', end='2024-12-31', interval='1d', auto_adjust=True)
df.to_parquet('data/raw/prices_daily.parquet')
```

### 1.2 Macro data (monthly / quarterly)

* Use **FRED** API for economic series (CPI, unemployment, yields). ([FRED][4])

Example (conceptual):

```python
from fredapi import Fred
fred = Fred(api_key='YOUR_KEY')
cpi = fred.get_series('CPIAUCSL', start_date='2000-01-01')
```

### 1.3 Company filings & fundamentals

* Use **SEC EDGAR** REST APIs / developer resources to fetch 10-K/10-Q/XBRL data (or use a wrapper). ([SEC][5])

### 1.4 News / earnings transcripts / alternative data

* Use **Finnhub** free tier for company news, earnings and some transcripts (has a free developer tier). ([finnhub.io][6])
* For search interest signals, use **Google Trends** via `pytrends` (unofficial wrapper). ([PyPI][7])

### 1.5 Historical dataset shortcuts

* If you prefer a ready dataset for prototyping, download Kaggle stock-price collections (many free datasets). ([kaggle.com][8])

**Deliverable:** `data/raw/prices_daily.parquet`, `data/raw/macro.parquet`, `data/raw/filings/`, `data/raw/news/`.

---

## Step 2 — Preprocessing & alignment (`src/preprocess/`)

Implement deterministic scripts that:

1. Convert daily → weekly (calendar week or business-week close) OHLCV.
2. Fill / hold-forward fundamentals to weekly timestamps (timestamp each report and hold until next).
3. Tag each record with ingestion timestamps and source (to simulate realistic latency).
4. Save processed time series to `data/processed/{prices_weekly.parquet, fundamentals_weekly.parquet, news_weekly.parquet}`.

**Acceptance test:** weekly file contains for each ticker an indexed weekly row count ≈ (years \* 52). No NA in target fields (or properly marked).

---

## Step 3 — Feature engineering (`src/features/`)

Implement `compute_features.py` to produce per-ticker, per-week feature vectors.

**Baseline features (examples):**

* Returns: 1w, 4w, 12w log-returns.
* Volatility: realized vol over 4w, 12w.
* Momentum: cumulative return over 4w, 12w.
* Liquidity: avg daily volume (4w), turnover.
* Valuation/fundamental: P/E, EPS change vs prior quarter, time-since-last-report.
* Macro lags: CPI change (m-1), yield spread.
* Text: weekly aggregated embedding + counts of “earnings / guidance” events.

**Text pipeline:** encode each news/article with a sentence-transformers model and aggregate per week (mean or attention-weighted). Use `sentence-transformers` or a finance-tuned encoder for improved signal.

**Output:** `data/processed/features_weekly.parquet` with schema `{ticker, date, features..., target_12w}`.

---

## Step 4 — Labeling and splits

* Define label(s): `R_{t→t+12w}` (12-week forward return) and `direction = sign(R)`.
* Create non-overlapping or partially overlapping label strategies; document choice. For walk-forward CV prefer non-overlapping test windows to avoid leakage.
* Implement a walker: e.g., train on 2010–2016, validate 2017, test 2018; roll forward by 1 year.

**Acceptance test:** labels align (i.e., features at week T map to price at T+12w) — check no look-ahead.

---

## Step 5 — Baseline model implementation (`src/models/`)

Start with a simple, reproducible baseline then iterate.

**Suggested progression**

1. **Numeric baseline:** Gradient-boosted tree (LightGBM / XGBoost) on engineered numeric features. Quick to train and interpretable.
2. **Sequence model:** Temporal Transformer consuming `T=52` weekly steps, early-fusion of text embeddings (concatenate), regression head for 12-week return.

   * Example backbone hyperparams: 4 layers, 8 heads, hidden dim 256.
   * Loss: Huber for regression + optional BCE auxiliary for direction.

**Deliverables:** `models/baseline_gbm.py` (train/eval), `models/temporal_transformer.py` (pytorch).

**Acceptance test:** model training script produces `models/checkpoint.pt` and `outputs/predictions_test.csv` with columns `ticker,date,pred_return,pred_prob`.

---

## Step 6 — Walk-forward backtest (`src/backtest/`)

Implement a reproducible backtester:

* Input: `predictions_test.csv`.
* Strategy: long when `pred_return > θ_pos`, short when `< θ_neg`, cash otherwise. Use realistic friction: 10–50 bps TC, slippage model.
* Compute performance: CAGR, annualized Sharpe, max drawdown, turnover. Save `reports/perf_summary.json` and equity curve plot.

**Acceptance test:** backtest run end-to-end; generate `reports/` with metrics and plots.

---

## Step 7 — Explainability & validation

* Compute feature importance (SHAP for tree model; attention or gradient-based attributions for transformer).
* Save top-3 textual events that drove each positive/negative prediction (use LLM/extraction later).

**Acceptance test:** for 10 sample predictions produce an explanation JSON with numeric attributions and 1–3 human-readable textual cues.

---

## Step 8 — Packaging & reproducibility

* Add `Makefile` or CLI to run core flows:

  * `make ingest` → runs ingestion scripts.
  * `make features` → preprocess + feature engineering.
  * `make train` → train model(s).
  * `make backtest` → produce report.
* Add unit tests for critical functions (data shapes, date-alignment, label correctness) in `tests/`.

---

## Step 9 — Optional: add LLM/text fusion

* Add a module that uses a sentence-transformer to encode weekly news and fuse embeddings (early or cross-attention). Example model: `sentence-transformers/all-mpnet-base-v2` (or a finance-tuned encoder).
* Retrain transformer backbone with text input. Measure uplift vs numeric baseline.

---

## Where to obtain free input data — concrete examples & links

* **Yahoo Finance / yfinance** — convenient for prototyping daily adjusted OHLCV via the `yfinance` Python library. ([GitHub][1])
* **Alpha Vantage** — free API key for historical daily data and indicators (call limits). ([Alpha Vantage][2])
* **Stooq** — free downloadable historical market data (daily/hourly/5-min). Good for markets where Yahoo coverage is limited. ([Stooq][3])
* **FRED** — macroeconomic series via API (CPI, unemployment, yields). ([FRED][4])
* **SEC EDGAR** — company filings and XBRL extracted data via SEC developer APIs or community wrappers. ([SEC][5])
* **Finnhub** — free tier for company news, earnings calendar and some transcripts. ([finnhub.io][6])
* **Kaggle** — multiple historical stock datasets for quick prototyping (download via Kaggle CLI). ([kaggle.com][8])
* **Google Trends** — query interest series via `pytrends` (useful alternative signal). ([PyPI][7])

*(Implement connectors to 2–3 of the above and confirm cross-checks between sources for data quality.)*

---

## Testing & acceptance criteria (quick checklist for the agent)

* [ ] `data/raw/` contains downloaded files for at least 50 tickers and 10 years of daily prices.
* [ ] `data/processed/features_weekly.parquet` present and has `n_features >= 40`.
* [ ] Training script runs without errors and produces `models/checkpoint.*`.
* [ ] Backtest produces `reports/perf_summary.json` and equity curve PDF/PNG.
* [ ] Basic explainability: top-5 numeric features by importance and a textual cue for 10 sample predictions.

---

## Minimal timeline for a single development sprint (suggested tasks)

1. Repo + env + sample ingestion (yfinance) → obtain daily prices for 100 tickers.
2. Weekly aggregation + compute baseline numeric features + label 12-week returns.
3. Train LightGBM baseline and run walk-forward backtest.
4. Add news embeddings (sentence-transformers) and test late-fusion uplift.

---

## Deliverables you should expect from the agent (explicit)

* `README.md` with run instructions.
* Ingestion scripts (`src/ingestion/*.py`) with config file for tickers/date ranges.
* `data/processed/features_weekly.parquet` and `labels_weekly.parquet`.
* `src/models/train.py` and `src/models/predict.py`.
* `src/backtest/run_backtest.py` and `reports/` with metrics.
* Unit tests in `tests/`.

---

If you want, I can now:

* produce **ready-to-run starter code** for the ingestion step (yfinance + weekly aggregation) and `requirements.txt`, or
* draft a **walk-forward backtest specification** with exact train/validation/test window sizes and transaction-cost assumptions.

Which of these two shall I produce now?

[1]: https://github.com/ranaroussi/yfinance?utm_source=chatgpt.com "Download market data from Yahoo! Finance's API"
[2]: https://www.alphavantage.co/documentation/?utm_source=chatgpt.com "Alpha Vantage API Documentation"
[3]: https://stooq.com/db/h/?utm_source=chatgpt.com "Free Historical Market Data"
[4]: https://fred.stlouisfed.org/docs/api/fred/?utm_source=chatgpt.com "St. Louis Fed Web Services: FRED® API"
[5]: https://www.sec.gov/about/developer-resources?utm_source=chatgpt.com "Developer Resources"
[6]: https://finnhub.io/docs/api?utm_source=chatgpt.com "API Documentation | Finnhub - Free APIs for realtime stock, ..."
[7]: https://pypi.org/project/pytrends/?utm_source=chatgpt.com "pytrends"
[8]: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset?utm_source=chatgpt.com "Stock Market Dataset"

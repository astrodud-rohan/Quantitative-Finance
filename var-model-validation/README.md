# VaR Model Backtesting and Challenger Model — Equity Index Trading Book
 
A portfolio project demonstrating **Market Risk Model Validation** competency:
backtesting a production Value-at-Risk (VaR) model using the three standard
regulatory pillars (Kupiec, Christoffersen, Basel Traffic-Light), and
benchmarking it against challenger methodologies.
 
## Why this project exists
 
Market risk model validation is one of the most mechanical, recurring
functions in Model Risk Management — VaR models get backtested on a rolling
basis because regulators (via Basel) directly tie backtesting performance to
capital requirements. This project demonstrates the exact statistical
machinery a market risk MRM analyst runs day to day, and — like Project 1 —
preserves a genuine, discovered methodological weakness rather than
engineering away the interesting parts.
 
## Key result
 
The production-style **Historical Simulation (HS) VaR model fails Kupiec's
unconditional coverage test** (28 breaches observed vs. 12.5 expected over a
1,250-day backtest) and **is currently sitting in the Basel Red Zone**
(11 breaches in the trailing 250 days), which triggers the maximum capital
multiplier add-on. A Filtered Historical Simulation (FHS) challenger model
passes every backtest and is recommended as the remediation path.
 
| # | Severity | Finding |
|---|----------|---------|
| 1 | **High** | Champion model significantly understates tail risk (Kupiec rejects, 28 vs 12.5 expected breaches) |
| 2 | Medium | Parametric alternative shows statistically significant breach clustering — not a viable remediation path |
| 3 | **High** | Champion model currently in Basel Red Zone — maximum capital multiplier add-on in effect |
 
**Validation conclusion: Not Approved for Continued Unconditional Use — Remediation Required.**
 
## Project structure
 
```
var-model-validation/
├── data/
│   ├── raw/                  # synthetic daily equity index prices/returns
│   └── processed/            # VaR estimates per model, per day
├── notebooks/
│   ├── 01_data_and_returns.ipynb         # return distribution, vol clustering
│   ├── 02_var_models.ipynb               # builds all 4 VaR models
│   └── 03_backtesting_validation.ipynb   # the real deliverable — backtesting
├── src/
│   ├── generate_market_data.py    # GARCH(1,1) + fat tails + injected stress regime
│   ├── var_models.py              # HS, Parametric, EWMA, FHS VaR
│   ├── backtesting.py             # Kupiec, Christoffersen, Basel traffic-light
│   └── plots.py
├── reports/
│   ├── VaR_Model_Backtesting_Report.docx   # flagship deliverable
│   └── *.png                               # exhibit charts referenced in the report
└── requirements.txt
```
 
## A note on the data
 
This project uses a **synthetic daily price series generated from a
GARCH(1,1) process with Student-t innovations** (for realistic fat tails)
and a **deliberately injected ~14-week volatility stress regime** partway
through the sample. This was a design choice, not an accident: it creates a
realistic scenario in which a slow-moving 250-day Historical Simulation VaR
model will lag a genuine regime change — exactly the kind of behavior real
VaR backtesting is designed to catch. Swap in a real price series (any
liquid index, FX pair, or single-name equity) at
`data/raw/index_prices.csv` with matching column names and the pipeline
runs unchanged.
 
## How to run
 
```bash
pip install -r requirements.txt
 
python src/generate_market_data.py
python src/var_models.py
python src/backtesting.py
python src/plots.py
 
# Or step through the notebooks in order
jupyter notebook notebooks/01_data_and_returns.ipynb
```
 
## Methodology references
 
- Kupiec (1995), "Techniques for Verifying the Accuracy of Risk Measurement Models"
- Christoffersen (1998), "Evaluating Interval Forecasts"
- Basel Committee on Banking Supervision, Internal Models Approach — backtesting framework (traffic-light approach)
- RiskMetrics (J.P. Morgan, 1996) — EWMA volatility methodology
- Filtered Historical Simulation — Barone-Adesi, Giannopoulos & Vosper (1999)
 
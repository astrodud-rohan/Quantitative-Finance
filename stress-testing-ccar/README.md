# CCAR-Style Stress Testing Model Validation

A portfolio project demonstrating **stress testing model validation**
competency: building a CCAR-style credit loss satellite model, projecting
losses under Fed-style macroeconomic scenarios, and independently
validating the model's behavior — especially its behavior in scenarios
more severe than anything in its own training history.

## Why this project exists

CCAR validation is a distinct skill from backtesting (Projects 1-2):
instead of checking a model against history that already happened, you're
assessing whether a model can be trusted in scenarios that **haven't**
happened yet. The central validation question shifts from "did this match
reality?" to "how much should we trust this extrapolation?" — and that
distinction is exactly what this project is built to demonstrate.

## Key result

The satellite model fits its own historical data extremely well
(R² = 0.986), but two real problems emerge under independent validation:

1. **Severe multicollinearity** (-0.95 correlation) between unemployment
   and HPI growth in the historical training data destabilizes individual
   coefficient estimates and undermines confidence in the model under
   scenarios where these variables decouple from their historical
   relationship.
2. **The Severely Adverse scenario requires extrapolation** beyond the
   model's historical training range in 8 of 9 stress horizon quarters —
   meaning most of the loss projection in the scenario that matters most
   for capital adequacy relies on the model holding far outside any data
   it was ever fit on.

A sensitivity analysis further shows that a defensible alternative
specification (a quadratic unemployment term) produces **14.5% higher**
cumulative losses specifically in the Severely Adverse scenario — model
specification risk is smallest where the model is well-supported by data,
and largest exactly where extrapolation is heaviest.

| # | Severity | Finding |
|---|----------|---------|
| 1 | **High** | Severe multicollinearity (-0.95) between unemployment and HPI growth destabilizes coefficients |
| 2 | **High** | Severely Adverse scenario requires extrapolation beyond training range in 8 of 9 quarters |
| 3 | Medium | Specification risk (linear vs. quadratic) concentrated in the Severely Adverse scenario (+14.5%) |

**Validation conclusion: Approved with Conditions.**

## Project structure

```
stress-testing-ccar/
├── data/
│   ├── raw/                  # synthetic 10-year quarterly macro + loss history
│   └── processed/             # scenario projections, sensitivity results
├── notebooks/
│   ├── 01_macro_history_and_satellite_model.ipynb
│   ├── 02_scenario_projection.ipynb
│   └── 03_sensitivity_and_validation.ipynb
├── src/
│   ├── generate_macro_history.py   # synthetic macro/loss data with one embedded downturn
│   ├── satellite_model.py          # the macro -> loss regression
│   ├── scenario_engine.py          # Fed-style Baseline/Adverse/Severely Adverse scenarios
│   ├── stress_projection.py        # projects losses; flags extrapolation
│   ├── sensitivity_analysis.py     # lag/functional-form specification risk
│   └── plots.py
├── reports/
│   ├── CCAR_Stress_Testing_Validation_Report.docx   # flagship deliverable
│   └── *.png                                        # exhibit charts
└── requirements.txt
```

## A note on the data

This project uses **synthetic quarterly macro and loss data spanning 10
years**, deliberately constructed so the historical record includes only
one moderate downturn (unemployment peaking at 8.13%) — never a severe
one. This is realistic: many banks' own loss histories don't span a truly
severe recession, which is exactly why CCAR satellite models so often face
genuine extrapolation risk when Fed Severely Adverse scenarios are applied.
Swap in real historical macro/loss data at
`data/raw/macro_and_loss_history.csv` with matching columns and the
pipeline runs unchanged.

## How to run

```bash
pip install -r requirements.txt

python src/generate_macro_history.py
python src/satellite_model.py
python src/scenario_engine.py
python src/stress_projection.py
python src/sensitivity_analysis.py
python src/plots.py

# Or step through the notebooks in order
jupyter notebook notebooks/01_macro_history_and_satellite_model.ipynb
```

## Methodology references

- Federal Reserve CCAR/DFAST supervisory scenario framework
- SR 11-7: Guidance on Model Risk Management — conceptual soundness and
  outcomes analysis under extrapolated conditions
- Standard credit risk literature on convexity in loss-rate response to
  unemployment (loss rates often accelerate disproportionately at higher
  unemployment levels, motivating the quadratic specification test)

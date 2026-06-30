# Independent Validation of a Retail Credit PD Model
 
A portfolio project demonstrating end-to-end **Model Risk Management (MRM)**
competency: building a credit risk Probability of Default (PD) scorecard,
then independently validating it against industry-standard frameworks
(SR 11-7 principles, Basel/IRB validation practice).
 
## Why this project exists
 
Most data science portfolios show *model building*. Model Risk Analyst roles
are about *model challenging* — independently re-deriving results, finding
where a model's assumptions break, and writing a risk-rated opinion a credit
committee or regulator can rely on. This project is structured to mirror that
real workflow, including a genuine methodological flaw that was discovered
during the build and preserved as a documented finding rather than quietly
fixed (see Finding 1 below).
 
## Key result
 
During independent validation, **WoE quantile binning was found to collapse
two sparse delinquency-count variables into a single bin**, zeroing their
model coefficients and materially weakening the model's discriminatory
power. This is the kind of subtle, high-impact issue real model validators
are paid to find — and is the centerpiece finding of the Independent
Validation Report.
 
| # | Severity | Finding |
|---|----------|---------|
| 1 | **High** | WoE binning collapses sparse delinquency-count variables, zeroing two coefficients |
| 2 | Medium | Discriminatory power (Gini) below internal benchmark on all samples |
| 3 | Medium | Statistically significant calibration deterioration in top risk deciles (Hosmer-Lemeshow p = 0.0029) |
| 4 | Low | PSI stable but observed default rate rising — a monitoring gap |
 
**Validation conclusion: Approved with Conditions.**
 
## Project structure
 
```
pd-model-validation/
├── data/
│   ├── raw/                  # synthetic credit dataset (GMSC-schema)
│   └── processed/            # WoE-transformed train/test/OOT splits, scored datasets
├── notebooks/
│   ├── 01_data_prep.ipynb               # cleaning, splitting, WoE binning
│   ├── 02_model_development.ipynb       # "development team" — fits the scorecard
│   └── 03_independent_validation.ipynb  # "MRM team" — the real deliverable
├── src/
│   ├── generate_synthetic_data.py
│   ├── data_processing.py
│   ├── model_training.py
│   ├── validation_metrics.py    # Gini, KS, PSI, CSI, Hosmer-Lemeshow, binomial backtest
│   ├── challenger_model.py
│   └── plots.py
├── reports/
│   ├── Model_Development_Document.docx     # the "handoff" doc from dev team
│   ├── Independent_Validation_Report.docx  # flagship deliverable
│   └── *.png                               # exhibit charts referenced in the IVR
└── requirements.txt
```
 
## A note on data
 
This project uses a **synthetic dataset generated to match the schema,
correlations, and ~6.6% default rate of the real "Give Me Some Credit"
(Kaggle) dataset**, with a built-in 24-month time dimension so an
out-of-time (OOT) holdout could be constructed exactly as a real validator
would require. Swap in the real Kaggle CSV at `data/raw/credit_data.csv`
with matching column names and the entire pipeline runs unchanged.
 
## How to run
 
```bash
pip install -r requirements.txt
 
# 1. Generate data and run the pipeline
python src/generate_synthetic_data.py
python src/data_processing.py
python src/model_training.py
python src/validation_metrics.py
python src/challenger_model.py
python src/plots.py
 
# 2. Or step through the notebooks in order
jupyter notebook notebooks/01_data_prep.ipynb
```
 
## Methodology references
 
- Federal Reserve / OCC **SR 11-7**: Guidance on Model Risk Management
- Basel Committee internal ratings-based (IRB) validation standards
- Hosmer-Lemeshow goodness-of-fit test for calibration
- Population/Characteristic Stability Index (PSI/CSI) for distributional drift
- Binomial backtesting for per-grade default rate validation
# Quantitative Finance Projects
 
Five end-to-end projects covering the core competencies of a Model Risk Management function: independent model validation, regulatory backtesting, governance and tiering, stress testing, and ongoing monitoring. Each project includes runnable Python code, Jupyter notebooks, and SR 11-7-style validation documentation.
 
---
 
## Project 1: Independent Validation of a Credit Risk PD Model
 
### What is it about
Builds a retail credit Probability of Default (PD) scorecard, then independently validates it the way a second-line Model Risk Management function would — re-deriving every statistic rather than trusting the development team's numbers. A genuine methodological flaw discovered during the build (not engineered for effect) became the centerpiece finding.
 
### Core concepts
- Weight-of-Evidence (WoE) binning and scorecard development
- Discriminatory power testing (Gini, KS, AUC) on train/test/out-of-time samples
- Calibration testing (Hosmer-Lemeshow, decile-level predicted vs. actual)
- Population/Characteristic Stability Index (PSI/CSI)
- Challenger model benchmarking
- Risk-rated findings (High/Medium/Low) in an SR 11-7-style Independent Validation Report
### Tech stack used
Python (pandas, NumPy, scikit-learn, SciPy, Matplotlib), Jupyter, Node.js `docx` library for Word report generation
 
### Result or Findings
Quantile-based WoE binning collapsed two sparse delinquency-count variables into a single bin, zeroing their coefficients and materially weakening the model's Gini (0.28–0.30, below the 0.40 internal benchmark). Calibration was also found to deteriorate significantly in the highest-risk deciles (Hosmer-Lemeshow p = 0.0029).
 
### What this project achieved
Demonstrated the full independent validation workflow end to end — conceptual soundness review, quantitative testing, benchmarking, and a formally risk-rated findings report — culminating in an "Approved with Conditions" validation opinion with specific, evidence-backed remediation conditions.
 
---
 
## Project 2: VaR Model Backtesting and Challenger Model
 
### What is it about
Backtests a production-style Value-at-Risk (VaR) model for an equity trading book using the three standard regulatory pillars, and benchmarks it against three challenger methodologies. Built on synthetic market data with a deliberately injected volatility stress regime to create realistic backtesting breaches.
 
### Core concepts
- Historical Simulation, Parametric, EWMA, and Filtered Historical Simulation VaR
- Kupiec Proportion of Failures (POF) test — unconditional coverage
- Christoffersen Independence test — breach clustering detection
- Basel Traffic-Light approach and regulatory capital multiplier implications
- GARCH(1,1) volatility modeling for realistic synthetic market data
### Tech stack used
Python (pandas, NumPy, SciPy, Matplotlib), Jupyter, Node.js `docx` library for Word report generation
 
### Result or Findings
The champion Historical Simulation model produced 28 breaches against an expected 12.5 over a 1,250-day window (Kupiec p = 0.0002) and is currently sitting in the Basel Red Zone — triggering the maximum capital multiplier add-on. A Filtered Historical Simulation challenger passed every backtest.
 
### What this project achieved
Showed the mechanical difference between a model that's miscalibrated (Kupiec) versus one that fails specifically during stress (Christoffersen), and tied both directly to a real, quantified capital cost — connecting statistical findings to business and regulatory consequences, not just p-values.
 
---
 
## Project 3: Model Risk Tiering and Inventory Framework
 
### What is it about
Designs a defensible, weighted model risk tiering methodology consistent with SR 11-7's risk-based validation principle, applies it to a realistic 18-model inventory spanning credit, market, compliance, and operational risk, and reports the results to a Model Risk Committee. The least code-heavy, most governance-focused project in the portfolio.
 
### Core concepts
- Multi-factor weighted scoring (materiality, complexity, usage/reliance, data quality risk)
- Risk-based allocation of finite validation resources
- Tier-to-validation-cadence mapping (annual / 18-month / 36-month cycles)
- Model inventory governance and committee reporting
- Auditable, formula-driven (not hardcoded) scoring workbook
### Tech stack used
Python (pandas, openpyxl), Node.js `docx` library for Word report generation, Excel (live formulas, conditional formatting)
 
### Result or Findings
Of 18 models, 7 were tiered High, 8 Medium, and 3 Low, implying roughly 13 validations due per year, concentrated in the Finance and Markets functions — a concentration the governance report flagged as a validation-capacity planning risk rather than passing over silently.
 
### What this project achieved
Demonstrated governance judgment that's largely absent from typical data science portfolios: the ability to design and defend a prioritization framework under finite resources, not just execute statistical tests on an already-given model.
 
---
 
## Project 4: CCAR-Style Stress Testing and Sensitivity Analysis
 
### What is it about
Builds a CCAR-style credit loss "satellite model" linking macroeconomic variables to portfolio losses, projects losses under Fed-style Baseline/Adverse/Severely Adverse scenarios over a 9-quarter horizon, and validates the model's behavior specifically in scenarios more severe than its own training history.
 
### Core concepts
- Satellite model regression (macro variables → portfolio loss rate)
- Fed-style CCAR/DFAST scenario construction
- Extrapolation risk detection — scenario inputs vs. historical training range
- Multicollinearity diagnosis and its effect on coefficient stability
- Specification sensitivity testing (lag structure, functional form)
### Tech stack used
Python (pandas, NumPy, scikit-learn, Matplotlib), Jupyter, Node.js `docx` library for Word report generation
 
### Result or Findings
Severe multicollinearity (-0.95 correlation) between unemployment and HPI growth destabilized individual coefficients, and the Severely Adverse scenario required extrapolation beyond the model's historical training range in 8 of 9 quarters. A quadratic specification produced 14.5% higher losses specifically in that scenario.
 
### What this project achieved
Captured a validation question distinct from backtesting: not "did this match reality" but "how much should we trust this model outside any data it's ever seen" — and quantified that uncertainty rather than leaving it as a qualitative caveat.
 
---
 
## Project 5: Ongoing Monitoring Dashboard for a Deployed Model
 
### What is it about
Simulates 12 months of production scoring data for a deployed credit model and builds the live monitoring infrastructure — an interactive dashboard, a thresholds policy, and an escalation report — that operates in the gap between full annual validations, where most of a model risk function's routine attention actually goes.
 
### Core concepts
- Population Stability Index (PSI) monitoring against a development baseline
- Discriminatory power decay tracking (Gini retained vs. baseline)
- Override rate as a leading indicator of eroding model trust
- RAG (Red/Amber/Green) threshold design and "weakest link" escalation logic
- Root cause assessment (model staleness vs. population shift)
### Tech stack used
Python (pandas, NumPy, scikit-learn, Matplotlib), vanilla HTML/CSS/JavaScript (dependency-free Canvas charts), Node.js `docx` library for Word report generation
 
### Result or Findings
Discriminatory power eroded from 123% to 74% of baseline and the override rate rose from 2% to 13% over 12 months — while PSI remained Green throughout, directly demonstrating that population stability alone is an incomplete early-warning signal.
 
### What this project achieved
Closed the model lifecycle loop left open by Projects 1–4: showed that validation is not a once-a-year event, and built the actual artifact (a live dashboard, not just a report) that a model risk analyst would use day to day to catch deterioration early.
 
---
 
## How the five projects connect
 
A finding from Project 2 (a stable VaR score distribution masking a rising actual breach rate) reappears independently in Project 5 (a stable PSI masking declining discriminatory power) — the same underlying lesson about population-stability metrics surfacing twice, from two different model types, validated two different ways. Together, the five projects span the full model risk lifecycle: build and validate (1), backtest against history (2), prioritize and govern (3), stress beyond history (4), and monitor continuously after deployment (5).
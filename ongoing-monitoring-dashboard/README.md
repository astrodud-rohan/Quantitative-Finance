# Ongoing Monitoring Dashboard for a Deployed Credit Model

A portfolio project demonstrating **post-deployment model monitoring**
competency: the work that happens in the gaps between full annual
validations, where most of a model risk function's day-to-day
attention actually goes.

## Why this project exists

Projects 1-4 in this portfolio are all about validating a model **at a
point in time**. In reality, models are validated once a year (or less
often) but used every day in between — and a lot can go wrong in that
gap. This project simulates 12 months of production data for a deployed
credit scorecard and builds the actual monitoring infrastructure (an
interactive dashboard, a thresholds policy, and an escalation report)
that catches deterioration before the next scheduled validation.

## Key result

Three monitoring metrics were tracked monthly against RAG thresholds:

| Metric | Month 1 | Month 12 | Status by Month 12 |
|---|---|---|---|
| PSI (population stability) | 0.0008 | 0.0897 | **Green** throughout |
| Gini retained vs. baseline | 122.9% | 73.9% | **Red** |
| Override rate | 2.0% | 13.1% | **Red** |

The model's discriminatory power eroded steadily over the year while its
score distribution remained stable (Green PSI throughout) — a direct,
concrete illustration of why population stability alone is an
insufficient early-warning signal, and why this framework deliberately
tracks three independent metrics rather than one composite score.

By Month 12, the model breaches Red status on two of three metrics,
triggering a formal escalation report to the Model Risk Committee.

## What's inside

- **An interactive HTML dashboard** (`dashboard/monitoring_dashboard.html`)
  — a fully self-contained, dependency-free control-room style dashboard
  with live RAG indicators, monthly trend charts, and threshold lines.
  Open it directly in any browser.
- **A policy document** defining the three monitored metrics, their
  thresholds, and required actions by RAG status
- **An escalation report** — the actual document Model Risk Management
  would produce and present to committee when Red status is triggered

## Project structure

```
ongoing-monitoring-dashboard/
├── src/
│   ├── simulate_production_data.py   # 12 months of drifting production scores
│   ├── monitoring_metrics.py         # PSI, Gini decay, override rate, RAG logic
│   └── plots.py
├── dashboard/
│   └── monitoring_dashboard.html     # the interactive deliverable (open in any browser)
├── docs/
│   └── Ongoing_Monitoring_Framework.docx   # thresholds and escalation policy
├── reports/
│   ├── Monthly_Monitoring_Report_Month12.docx   # the escalation report
│   └── *.png                                    # exhibit charts
└── requirements.txt
```

## A note on the data

This project simulates 12 months of production scoring data for the
same conceptual credit PD scorecard validated in Project 1, with three
deliberately engineered degradation patterns: gradual population drift,
slow discrimination decay, and a rising override rate — building to a
realistic escalation event in the final months. Swap in real monthly
production scoring data with matching columns and the pipeline runs
unchanged.

## How to run

```bash
pip install -r requirements.txt

python src/simulate_production_data.py
python src/monitoring_metrics.py
python src/plots.py

# Open the dashboard directly in a browser - no server needed
open dashboard/monitoring_dashboard.html   # macOS
# or just double-click the file
```

## Methodology references

- SR 11-7: Guidance on Model Risk Management — ongoing monitoring as a
  distinct activity from periodic validation
- Standard credit scorecard monitoring metrics: PSI, CSI, Gini decay,
  override/exception rate tracking

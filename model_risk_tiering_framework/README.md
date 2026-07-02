# Model Risk Tiering and Inventory Framework
 
A portfolio project demonstrating **Model Risk Management governance**
competency: designing a defensible model risk tiering methodology
consistent with SR 11-7 principles, applying it to a realistic 18-model
inventory, and reporting the results to a Model Risk Committee.
 
## Why this project exists
 
Most data science portfolios show statistical model-building or
validation. This project demonstrates the other half of the job that's
just as important and far less commonly shown: **governance judgment**.
Model Risk Management isn't only about running Kupiec tests or PSI
calculations — it's about deciding, defensibly and consistently, which
models deserve the most validation attention when resources are finite.
This is the SR 11-7 principle of risk-based validation prioritization,
operationalized.
 
## What's inside
 
- A **weighted scoring methodology** across four dimensions (materiality,
  complexity, usage/reliance, data quality risk), with every weight and
  threshold documented and justified
- An **18-model inventory** spanning credit, market, liquidity,
  compliance, operational risk, and supporting business functions —
  each scored with a written, defensible rationale (not randomly
  generated)
- A **live Excel workbook** with formulas (not hardcoded values) so the
  scoring is auditable and the workbook stays dynamic if scores are
  revised
- A **governance report** with real, substantive observations (e.g.,
  validation capacity concentration risk in one business function) —
  not just a restatement of the inventory
## Project structure
 
```
model-risk-tiering-framework/
├── src/
│   ├── tiering_engine.py     # the scoring/tiering logic (also mirrored as live Excel formulas)
│   └── plots.py
├── scripts/
│   ├── build_inventory.py        # the 18-model inventory with documented scoring rationale
│   ├── build_inventory_xlsx.py   # builds the live Excel workbook
│   ├── generate_methodology_doc.js
│   └── generate_governance_report.js
├── inventory/
│   └── Model_Inventory.xlsx      # the live, auditable inventory workbook
├── docs/
│   └── Model_Tiering_Methodology.docx   # the policy document defining the framework
├── reports/
│   ├── Model_Inventory_Governance_Report.docx  # flagship deliverable — committee report
│   └── *.png                                   # exhibit charts
└── requirements.txt
```
 
## Key result
 
| Tier | Count | % | Validation Frequency | Monitoring |
|------|-------|---|----------------------|------------|
| Tier 1 (High) | 7 | 39% | Annual | Monthly |
| Tier 2 (Medium) | 8 | 44% | Every 18 months | Quarterly |
| Tier 3 (Low) | 3 | 17% | Every 36 months | Semi-annual |
 
Estimated validation workload: **13 models per year**, concentrated in
Finance and Markets functions — a concentration the governance report
flags as a capacity-planning risk worth confirming with the validation
team, rather than a problem to silently note and move past.
 
## How to run
 
```bash
pip install -r requirements.txt
 
python scripts/build_inventory.py          # builds and scores the inventory (CSV)
python scripts/build_inventory_xlsx.py     # builds the live Excel workbook
python src/plots.py                        # generates governance report exhibits
 
# Recalculate Excel formulas and verify zero errors (requires LibreOffice)
python /mnt/skills/public/xlsx/scripts/recalc.py inventory/Model_Inventory.xlsx
```
 
## Methodology references
 
- Federal Reserve / OCC **SR 11-7**: Guidance on Model Risk Management —
  risk-based prioritization of validation resources
- Basel Committee model governance principles
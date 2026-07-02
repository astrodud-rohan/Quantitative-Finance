"""
Constructs a representative 15-model inventory spanning credit, market, operational, compliance,
and pricing risk.
"""

import sys
sys.path.append("src")
import pandas as pd
from tiering_engine import score_inventory

MODELS = [
    ("Retail Credit PD Scorecard", "Logistic regression (WoE scorecard)", "Credit Risk", "Retail Banking", 4, 2, 4, 2,
     "Feeds both credit decisioning and regulatory capital (IRB); moderate complexity (interpretable scorecard);"
     "decisions are largely automated below a referral threshold; data is internal bureau/application data, well governed."
    ),
    ("Wholesale LGD Model", "Regression on workout recovery data", "Credit Risk", "Corporate Banking", 4, 3, 3, 3,
     "Directly drives loss provisioning and regulatory capital; moderate-high complexity due to segment-specific "
     "recovery curves; used as a strong input but with analyst overlay; recovery data is sparse and judgmental."
    ),
    ("Market Risk VaR Model (Equity/FX Trading Book)", "Historical Simulation VaR", "Market Risk", "Markets", 5, 3, 5, 2,
     "Directly determines regulatory trading capital requirements; moderate complexity; fully automated daily "
     "capital calculation with no human override; relies on well-governed internal trade and price data."
    ),
    ("IFRS 9 / CECL Expected Credit Loss Model", "PD x LGD x EAD lifetime ECL engine", "Credit Risk", "Finance", 5, 4, 5, 3,
     "Directly drives the loan loss allowance on the balance sheet and reported earnings; high complexity "
     "(multiple sub-models combined with macro scenario overlays); output flows almost automatically into "
     "financial statements; depends on forward-looking macro data with inherent uncertainty."
    ),
    ("AML Transaction Monitoring Model", "Rules engine + anomaly detection scoring", "Compliance", "Financial Crime", 5, 4, 4, 4,
     "False negatives carry severe regulatory and reputational consequences (sanctions, fines); hybrid "
     "rules/ML complexity; alerts drive investigation decisions with analyst review (not fully automated); "
     "depends on noisy, high-volume, frequently-changing transaction data."
    ),
    ("Fraud Detection Model (Card Transactions)", "Gradient boosting classifier", "Compliance", "Retail Banking", 4, 5, 5, 4,
     "Direct customer and financial impact from false positives/negatives; high complexity (black-box "
     "gradient boosting, hard to explain individual declines); near-real-time automated decisioning with "
     "minimal human override; depends on fast-evolving fraud patterns and third-party data feeds."
    ),
    ("CCAR / Stress Testing Credit Loss Model", "Satellite regression on macro scenarios", "Credit Risk", "Finance", 5, 4, 4, 3,
     "Directly informs capital planning and is subject to direct regulatory/supervisory scrutiny (CCAR); "
     "high complexity due to macro scenario linkage; results heavily inform but do not single-handedly "
     "determine capital actions; relies on macro data with structural uncertainty over stress horizons."
    ),
    ("Interest Rate Risk in the Banking Book (IRRBB) Model", "Cash flow / duration gap model", "Market Risk", "Treasury", 4, 3, 3, 2,
     "Informs balance sheet hedging and capital adequacy assessment; moderate complexity (behavioral "
     "assumptions on non-maturity deposits); used as a key input to ALCO decisions with committee judgment; "
     "data is internal balance sheet data, well governed."
    ),
    ("Liquidity Stress Testing Model", "Cash flow runoff model under stress scenarios", "Liquidity Risk", "Treasury", 4, 3, 3, 2,
     "Directly informs the bank's liquidity buffer and contingency funding plan; moderate complexity "
     "(runoff assumptions by deposit/funding type); informs ALCO decisions with judgment overlay; "
     "internal funding data, well governed."
    ),
    ("Operational Risk Capital Model (AMA-style)", "Loss distribution / scenario-based simulation", "Operational Risk", "Finance", 3, 4, 3, 4,
     "Contributes to regulatory capital but typically a smaller capital component than credit/market risk; "
     "high complexity (statistical loss distribution modeling, scenario blending); informs capital "
     "calculation with significant expert judgment overlay; relies on sparse internal loss data plus subjective scenario inputs."
    ),
    ("Derivative Pricing Model (Interest Rate Swaps)", "Discounted cash flow / curve-based pricing", "Market Risk", "Markets", 4, 3, 5, 2,
     "Directly determines trade valuation and daily P&L; moderate complexity (standard curve-based pricing, "
     "well-established methodology); fully automated valuation with no manual override in normal conditions; "
     "relies on well-governed, actively quoted market data."
    ),
    ("Customer Churn Prediction Model", "Random forest classifier", "Marketing Analytics", "Retail Banking", 2, 4, 2, 3,
     "Limited direct financial impact (informs retention marketing spend only); moderately high complexity "
     "(ensemble model, not easily interpretable;) used as one input among several in campaign targeting, "
     "with marketing judgment applied; relies on internal CRM data of moderate quality."
    ),
    ("Economic Capital Allocation Model", "Multi-risk-type aggregation model", "Enterprise Risk", "Finance", 4, 4, 3, 3,
     "Informs internal capital allocation and risk appetite, but is an internal management tool rather than "
     "a primary regulatory capital driver; high complexity (aggregates credit/market/operational risk with "
     "correlation assumptions); used as a key input to capital planning discussions with senior management "
     "judgment; depends on outputs of multiple upstream models, compounding data risk."
    ),
    ("Algorithmic Trading Execution Model", "Rules-based execution algorithm", "Market Risk", "Markets", 3, 3, 5, 2,
     "Financial impact is real but bounded by position limits and kill-switches; moderate complexity "
     "(deterministic execution rules, not statistical/ML); fully automated execution with no per-trade "
     "human override; relies on well-governed real-time market data feeds."
    ),
    ("Robo-Advisor Portfolio Allocation Model", "Mean-variance optimization with risk-profiling questionnaire", "Wealth Management", "Wealth Management", 3, 2, 4, 2,
     "Direct impact on individual client portfolios, but limited to retail wealth clients rather than "
     "balance-sheet-wide exposure; low-moderate complexity (standard mean-variance optimization, "
     "well-documented methodology); largely automated rebalancing with limited advisor override; "
     "relies on internal client questionnaire and market data, both reasonably well governed."
    ),
    ("Branch Cash Demand Forecasting Model", "Time series (ARIMA) forecast of ATM/branch cash needs", "Operations", "Retail Banking", 1, 2, 2, 2,
     "Purely an operational efficiency tool (cash logistics); no credit, market, or capital impact; "
     "low complexity (standard time series forecasting); used as a planning input with branch manager "
     "discretion to override; relies on stable internal transaction history."
    ),
    ("Employee Attrition Prediction Model", "Logistic regression on HR data", "Human Resources", "Corporate Functions", 1, 2, 1, 2,
     "No financial, credit, or customer impact; informs HR retention conversations only; low complexity; "
     "purely advisory with HR business partner judgment applied to every case; relies on internal HR "
     "data of reasonable quality."
    ),
    ("Internal Expense Allocation Model", "Rules-based cost allocation across business units", "Finance", "Corporate Functions", 2, 1, 2, 1,
     "Affects internal management reporting and business unit profitability views, but not external "
     "financial statements or regulatory capital; very low complexity (deterministic allocation rules, "
     "not statistical); used for internal reporting with Finance review; relies on well-governed "
     "internal general ledger data."
    ),
]

COLUMNS = [
    "model_name", "model_type", "owner_function", "business_line", "materiality", "complexity", "usage_reliance", "data_quality_risk", "scoring_rationale"
]

def build_inventory_df():
    df = pd.DataFrame(MODELS, columns=COLUMNS)
    df.insert(0, "model_id", [f"MDL-{i+1:03d}" for i in range(len(df))])
    return df

if __name__ == "__main__":
    df = build_inventory_df()
    scored_df = score_inventory(df)
    scored_df.to_csv("inventory/model_inventory_scored.csv", index=False)
    print(f"Built inventory of {len(scored_df)} models.")
    print(scored_df["tier"].value_counts())
    print()
    print(scored_df[["model_id", "model_name", "weighted_score", "tier"]].to_string(index=False))
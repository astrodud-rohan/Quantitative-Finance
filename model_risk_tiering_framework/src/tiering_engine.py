"""
Implements the bank's model risk tiering methodology, consistent with SR 11-7 principles.

Each model is scored 1-5 on four dimensions:
    - Materiality (weight 40%)
    - Complexity (weight 25%)
    - Usage / Reliance (weight 25%)
    - Data Quality Risk (weight 10%)

Weighted Score and Tier List
    - Weighted Score >= 3.60          -> Tier 1 (High)
    - 2.40 <= Weighted Score < 3.60   -> Tier 2 (Medium)
    - Weighted Score < 2.40           -> Tier 3 (Low)
"""

import pandas as pd

WEIGHTS = {
    "materiality": 0.40,
    "complexity": 0.25,
    "usage_reliance": 0.25,
    "data_quality_risk": 0.10
}

TIER_THRESHOLDS = {
    "Tier 1 (High)": 3.60,
    "Tier 2 (Medium)": 2.40,
    "Tier 3 (Low)": 0.0
}

VALIDATION_REQUIREMENTS = {
    "Tier 1 (High)": {
        "validation_frequency": "Annual",
        "monitoring_frequency": "Monthly",
        "sign_off_required": "Model Risk Committee",
        "validation_depth": "Full independent validation (conceptual soundness, outcome analysis, benchmarking, sensitivity testing)"
    },
    "Tier 2 (Medium)": {
        "validation_frequency": "Every 18 months",
        "monitoring_frequency": "Quarterly",
        "sign_off_required": "Model Risk Committee",
        "validation_depth": "Full independent validation (reduced benchmarking scope permitted)"
    },
    "Tier 3 (Low)": {
        "validation_frequency": "Every 36 months",
        "monitoring_frequency": "Semi-annual",
        "sign_off_required": "Line of Business Risk Owner",
        "validation_depth": "Light-touch validation (conceptual soundness and outcome analysis only)"
    },
}

def compute_weighted_score(materiality, complexity, usage_reliance, data_quality_risk):
    return round(
        materiality * WEIGHTS["materiality"] +
        complexity * WEIGHTS["complexity"] +
        usage_reliance * WEIGHTS["usage_reliance"] +
        data_quality_risk * WEIGHTS["data_quality_risk"],
        2
    )

def assign_tier(weighted_score):
    if weighted_score >= TIER_THRESHOLDS["Tier 1 (High)"]:
        return "Tier 1 (High)"
    elif weighted_score >= TIER_THRESHOLDS["Tier 2 (Medium)"]:
        return "Tier 2 (Medium)"
    else:
        return "Tier 3 (Low)"

def score_model(materiality, complexity, usage_reliance, data_quality_risk):
    weighted_score = compute_weighted_score(materiality, complexity, usage_reliance, data_quality_risk)
    tier = assign_tier(weighted_score)
    requirements = VALIDATION_REQUIREMENTS[tier]
    
    return {
        "weighted_score": weighted_score,
        "tier": tier,
        **requirements
    }

def score_inventory(inventory_df: pd.DataFrame) -> pd.DataFrame:
    results = inventory_df.apply(
        lambda row: score_model(
            row["materiality"], row["complexity"], row["usage_reliance"], row["data_quality_risk"]
        ),
        axis=1,
        result_type="expand",
    )
    return pd.concat([inventory_df, results], axis=1)

if __name__ == "__main__":
    # Sanity Check
    test_cases = [
        (5, 4, 5, 3),  # Expected Tier 1
        (3, 3, 3, 2),  # Expected Tier 2
        (1, 1, 1, 1)   # Expected Tier 3
    ]
    for m, c, u, d in test_cases:
        result = score_model(m, c, u, d)
        print(f"Materiality: {m}, Complexity: {c}, Usage/Reliance: {u}, Data Quality Risk: {d} "
              f"=> Weighted Score: {result['weighted_score']}, Tier: {result['tier']}")
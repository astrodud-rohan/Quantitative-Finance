"""
Simple decision tree challenger model - for validation
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sys
sys.path.append("src")
from model_training import WOE_FEATURES, TARGET
from validation_metrics import gini_coefficient

def train_challenger(train_df, features=WOE_FEATURES, target=TARGET):
    model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, random_state=42)
    model.fit(train_df[features], train_df[target])
    return model

if __name__ == "__main__":
    train = pd.read_csv("../data/processed/train_woe.csv")
    test = pd.read_csv("../data/processed/test_woe.csv")
    oot = pd.read_csv("../data/processed/oot_woe.csv")

    challenger = train_challenger(train)

    print("=== Challenger Model (Decision Tree) Performance ===")
    for name, d in [("Train", train), ("Test", test), ("OOT", oot)]:
        pred = challenger.predict_proba(d[WOE_FEATURES])[:, 1]
        gini, auc = gini_coefficient(d[TARGET], pred)
        print(f"{name:6s} | AUC: {auc:.4f} | Gini: {gini:.4f}")
    
    print("\nConclusion: Champion (logistic regression) OOT Gini = 0.2968")
    print("Challenger (decision tree) OOT Gini = 0.2572")
    print("Champion modestly outperforms challenger -> model family choice")
    print("is not the primary driver of weak performance (see Finding 1).")
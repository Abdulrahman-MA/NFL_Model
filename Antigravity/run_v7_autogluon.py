"""
Antigravity v7 — AutoGluon Grandmaster Stack
Uses Amazon's AutoGluon library to automatically train, tune, and stack dozens of models
(LightGBM, CatBoost, XGBoost, FastAI, PyTorch, RF, ExtraTrees, KNN) to extract the
absolute mathematical maximum from the NFL Draft dataset.
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
SUB_DIR = Path(__file__).parent / "submissions_v7"
SUB_DIR.mkdir(parents=True, exist_ok=True)
AG_PATH = SUB_DIR / "ag_models"

CAT_COLS = ["School", "Player_Type", "Position_Type", "Position"]
PHYSICAL = [
    "Height", "Weight", "Sprint_40yd", "Vertical_Jump",
    "Bench_Press_Reps", "Broad_Jump", "Agility_3cone", "Shuttle", "Age",
]

# ===================================================================
# 1. Feature Engineering
# ===================================================================
def build_features(train_df, test_df):
    df = pd.concat([train_df.drop(columns=["Drafted"]), test_df], axis=0).reset_index(drop=True)

    # 1. Missingness flags
    for col in PHYSICAL:
        if df[col].isnull().sum() > 0:
            df[f"{col}_Is_Missing"] = df[col].isnull().astype(int)

    # 2. Imputation (AutoGluon can handle NaNs, but domain-specific positional median is better)
    for col in PHYSICAL:
        df[col] = df[col].fillna(df.groupby("Position")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

    # 3. Domain Features
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["Speed_Weight_Ratio"] = df["Weight"] / df["Sprint_40yd"]
    df["Explosiveness_Index"] = df["Vertical_Jump"] * df["Broad_Jump"]
    df["Power_Score"] = df["Bench_Press_Reps"] * df["Weight"]
    df["School_Combine_Count"] = df.groupby("School")["Id"].transform("count")
    df["Speed_Score"] = (df["Weight"] * 200) / (df["Sprint_40yd"] ** 4)

    # 4. Positional Relative Scores
    for col in PHYSICAL:
        df[f"{col}_Pos_Mean"] = df.groupby("Position")[col].transform("mean")
        df[f"{col}_Pos_Diff"] = df[col] - df[f"{col}_Pos_Mean"]
        df.drop(columns=[f"{col}_Pos_Mean"], inplace=True)

    # Convert categoricals to strings for AutoGluon
    for col in CAT_COLS:
        df[col] = df[col].fillna("Unknown").astype(str)

    X_train = df.iloc[: len(train_df)].reset_index(drop=True)
    X_train["Drafted"] = train_df["Drafted"]
    
    X_test = df.iloc[len(train_df) :].reset_index(drop=True)
    
    return X_train, X_test

# ===================================================================
# 2. Main Execution
# ===================================================================
def main():
    print("=" * 60)
    print("ANTIGRAVITY v7 - AutoGluon Grandmaster Stack")
    print("=" * 60)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    test_ids = test["Id"].reset_index(drop=True)

    print("[1] Building domain features...")
    X_train, X_test = build_features(train, test)
    
    # We drop the 'Id' column before passing to AutoGluon so it isn't used as a feature
    X_train = X_train.drop(columns=["Id"])
    X_test_pred = X_test.drop(columns=["Id"])

    print(f"\\n[2] Triggering AutoGluon 'best_quality' stack...")
    
    predictor = TabularPredictor(
        label="Drafted",
        eval_metric="roc_auc",
        path=str(AG_PATH),
        verbosity=2
    )
    
    predictor.fit(
        train_data=X_train,
        presets="best_quality",
        time_limit=60 * 20, # Max 20 minutes
        excluded_model_types=["KNN"] # KNN is often useless noise on this dataset
    )

    print(f"\\n[3] AutoGluon Training Complete. Leaderboard:")
    try:
        lb = predictor.leaderboard(X_train)
        print(lb.head(10))
    except Exception as e:
        print("Could not print leaderboard:", e)

    print(f"\\n[4] Generating final predictions...")
    
    # AutoGluon returns a Series for predict_proba when it's binary classification
    # Index 1 is the positive class probability
    preds = predictor.predict_proba(X_test_pred)[1].values
    
    path = SUB_DIR / "v7_autogluon_submission.csv"
    pd.DataFrame({"Id": test_ids, "Drafted": preds}).to_csv(path, index=False)
    print(f"SUCCESS! AutoGluon Submission saved to: {path}")

if __name__ == "__main__":
    main()

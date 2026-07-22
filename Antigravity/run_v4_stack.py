"""
Antigravity v4 — Heterogeneous Stacking
Builds an ensemble using CatBoost, LightGBM, XGBoost, and Random Forest
on the proven 36-feature baseline.
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
SUB_DIR = Path(__file__).parent / "submissions_v4"
SUB_DIR.mkdir(parents=True, exist_ok=True)

CAT_COLS = ["School", "Player_Type", "Position_Type", "Position"]
PHYSICAL = [
    "Height", "Weight", "Sprint_40yd", "Vertical_Jump",
    "Bench_Press_Reps", "Broad_Jump", "Agility_3cone", "Shuttle", "Age",
]

# The EXACT CatBoost params from the 0.844 LB submission
CB_PARAMS = {
    "iterations": 1183,
    "learning_rate": 0.010492928203031996,
    "depth": 4,
    "l2_leaf_reg": 4.5594175959311425,
    "min_data_in_leaf": 7,
    "verbose": 0,
    "allow_writing_files": False,
}

# LightGBM params (highly regularized for small data)
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.01,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 3.0,
    "colsample_bytree": 0.7,
    "subsample": 0.8,
    "n_estimators": 1500,
    "verbose": -1,
}

# XGBoost params (shallow trees, regularized)
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.01,
    "max_depth": 3,
    "min_child_weight": 5,
    "alpha": 0.1,
    "lambda": 3.0,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "n_estimators": 1500,
}

# ===================================================================
# Feature prep (Original 36)
# ===================================================================
def build_features(train_df, test_df):
    df = pd.concat([train_df.drop(columns=["Drafted"]), test_df], axis=0).reset_index(drop=True)

    for col in PHYSICAL:
        if df[col].isnull().sum() > 0:
            df[f"{col}_Is_Missing"] = df[col].isnull().astype(int)

    for col in PHYSICAL:
        df[col] = df[col].fillna(df.groupby("Position")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["Speed_Weight_Ratio"] = df["Weight"] / df["Sprint_40yd"]
    df["Explosiveness_Index"] = df["Vertical_Jump"] * df["Broad_Jump"]
    df["Power_Score"] = df["Bench_Press_Reps"] * df["Weight"]
    df["School_Combine_Count"] = df.groupby("School")["Id"].transform("count")
    df["Speed_Score"] = (df["Weight"] * 200) / (df["Sprint_40yd"] ** 4)

    for col in PHYSICAL:
        df[f"{col}_Pos_Mean"] = df.groupby("Position")[col].transform("mean")
        df[f"{col}_Pos_Diff"] = df[col] - df[f"{col}_Pos_Mean"]
        df.drop(columns=[f"{col}_Pos_Mean"], inplace=True)

    # For CatBoost: keep as category string
    # For others: factorize to integer
    df_cat = df.copy()
    for col in CAT_COLS:
        df_cat[col] = df_cat[col].fillna("Unknown").astype("category")
        df[col] = pd.factorize(df[col].fillna("Unknown"))[0]

    X_train_cb = df_cat.iloc[: len(train_df)].drop(columns=["Id"]).reset_index(drop=True)
    X_test_cb = df_cat.iloc[len(train_df) :].drop(columns=["Id"]).reset_index(drop=True)
    
    X_train_other = df.iloc[: len(train_df)].drop(columns=["Id"]).reset_index(drop=True)
    X_test_other = df.iloc[len(train_df) :].drop(columns=["Id"]).reset_index(drop=True)
    
    return X_train_cb, X_test_cb, X_train_other, X_test_other

# ===================================================================
# Training Functions
# ===================================================================
def train_catboost(X, X_test, y, n_folds=10, seed=42):
    print("\n[CatBoost] Training...")
    cat_indices = [i for i, c in enumerate(X.columns) if c in CAT_COLS]
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        p = CB_PARAMS.copy()
        p["random_seed"] = seed + fold
        model = CatBoostClassifier(**p, cat_features=cat_indices)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=(X.iloc[va_idx], y.iloc[va_idx]),
            early_stopping_rounds=50,
            use_best_model=True,
        )
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / n_folds

    auc = roc_auc_score(y, oof)
    print(f"CatBoost OOF AUC: {auc:.5f}")
    return oof, test_pred

def train_lightgbm(X, X_test, y, n_folds=10, seed=42):
    print("\n[LightGBM] Training...")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        p = LGB_PARAMS.copy()
        p["random_state"] = seed + fold
        model = lgb.LGBMClassifier(**p)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / n_folds

    auc = roc_auc_score(y, oof)
    print(f"LightGBM OOF AUC: {auc:.5f}")
    return oof, test_pred

def train_xgboost(X, X_test, y, n_folds=10, seed=42):
    print("\n[XGBoost] Training...")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        p = XGB_PARAMS.copy()
        p["random_state"] = seed + fold
        p["early_stopping_rounds"] = 50
        model = xgb.XGBClassifier(**p)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
            verbose=False,
        )
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / n_folds

    auc = roc_auc_score(y, oof)
    print(f"XGBoost OOF AUC: {auc:.5f}")
    return oof, test_pred

def train_rf(X, X_test, y, n_folds=10, seed=42):
    print("\n[RandomForest] Training...")
    # RF doesn't like NaNs, fill with massive negative
    X = X.fillna(-999)
    X_test = X_test.fillna(-999)
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        model = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=5, random_state=seed+fold, n_jobs=-1)
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / n_folds

    auc = roc_auc_score(y, oof)
    print(f"RandomForest OOF AUC: {auc:.5f}")
    return oof, test_pred

def rank01(v):
    return pd.Series(v).rank(pct=True, method="average").values

def main():
    print("=" * 60)
    print("ANTIGRAVITY v4 - Heterogeneous Stacking")
    print("=" * 60)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    y = train["Drafted"].reset_index(drop=True)
    test_ids = test["Id"].reset_index(drop=True)

    X_cb, X_test_cb, X_other, X_test_other = build_features(train, test)

    # Base Models
    oof_cb, test_cb = train_catboost(X_cb, X_test_cb, y)
    oof_lgb, test_lgb = train_lightgbm(X_other, X_test_other, y)
    oof_xgb, test_xgb = train_xgboost(X_other, X_test_other, y)
    oof_rf, test_rf = train_rf(X_other, X_test_other, y)

    # 1. Rank-based Equal Blend
    print("\n[Blending] Equal Rank Average...")
    blend_oof = (rank01(oof_cb) + rank01(oof_lgb) + rank01(oof_xgb) + rank01(oof_rf)) / 4
    blend_test = (rank01(test_cb) + rank01(test_lgb) + rank01(test_xgb) + rank01(test_rf)) / 4
    blend_auc = roc_auc_score(y, blend_oof)
    print(f"Equal Rank Blend OOF AUC: {blend_auc:.5f}")

    # 2. Logistic Regression Meta-Model (Stacking)
    print("\n[Stacking] Logistic Regression Meta-Model...")
    # Using raw probabilities for stacking
    OOF_matrix = np.column_stack([oof_cb, oof_lgb, oof_xgb, oof_rf])
    TEST_matrix = np.column_stack([test_cb, test_lgb, test_xgb, test_rf])
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    meta_oof = np.zeros(len(y))
    meta_test = np.zeros(len(test_ids))
    
    for fold, (tr_idx, va_idx) in enumerate(cv.split(OOF_matrix, y), 1):
        meta_model = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs')
        meta_model.fit(OOF_matrix[tr_idx], y.iloc[tr_idx])
        meta_oof[va_idx] = meta_model.predict_proba(OOF_matrix[va_idx])[:, 1]
        meta_test += meta_model.predict_proba(TEST_matrix)[:, 1] / 10
        
    stack_auc = roc_auc_score(y, meta_oof)
    print(f"Stacking OOF AUC: {stack_auc:.5f}")

    # 3. Best 3 Blend
    # Drop RF since it usually scores lower
    blend3_oof = (rank01(oof_cb) + rank01(oof_lgb) + rank01(oof_xgb)) / 3
    blend3_test = (rank01(test_cb) + rank01(test_lgb) + rank01(test_xgb)) / 3
    blend3_auc = roc_auc_score(y, blend3_oof)
    print(f"Top 3 Blend (CB+LGB+XGB) OOF AUC: {blend3_auc:.5f}")

    # Save outputs
    print("\nSaving submissions...")
    results = [
        ("v4_catboost", test_cb, roc_auc_score(y, oof_cb)),
        ("v4_lightgbm", test_lgb, roc_auc_score(y, oof_lgb)),
        ("v4_xgboost", test_xgb, roc_auc_score(y, oof_xgb)),
        ("v4_rf", test_rf, roc_auc_score(y, oof_rf)),
        ("v4_blend_all", blend_test, blend_auc),
        ("v4_blend_top3", blend3_test, blend3_auc),
        ("v4_stacked_lr", meta_test, stack_auc),
    ]

    for name, preds, auc in results:
        path = SUB_DIR / f"{name}.csv"
        pd.DataFrame({"Id": test_ids, "Drafted": preds}).to_csv(path, index=False)
        
    print("\nDone! Best models:")
    for name, _, auc in sorted(results, key=lambda x: x[2], reverse=True):
        print(f"  {name}: {auc:.5f}")

if __name__ == "__main__":
    main()

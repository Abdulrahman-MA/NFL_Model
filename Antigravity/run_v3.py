"""
Antigravity v3 — Conservative Approach Starting from LB-Proven Baseline.

Philosophy: Your old 36-feature CatBoost scored 0.844 on public LB.
The v2 with 80 features scored LOWER (0.840) despite higher OOF.
With only 2781 training samples, SIMPLICITY WINS.

Strategy:
1. Start with the EXACT 36-feature matrix that scored 0.844
2. Multi-seed it (reduce variance without adding features)
3. Add ONLY 2-3 safe features that are very robust
4. Test ablations — if adding features hurts, drop them
5. Simple equal-weight ensemble (no OOF-fitted weights)
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
SUB_DIR = Path(__file__).parent / "submissions_v3"
SUB_DIR.mkdir(parents=True, exist_ok=True)

CAT_COLS = ["School", "Player_Type", "Position_Type", "Position"]
PHYSICAL = [
    "Height", "Weight", "Sprint_40yd", "Vertical_Jump",
    "Bench_Press_Reps", "Broad_Jump", "Agility_3cone", "Shuttle", "Age",
]

# The EXACT params from the 0.844 LB submission
# NOTE: The old notebook did NOT specify loss_function or eval_metric.
# CatBoost defaults to Logloss for both, which produces well-calibrated
# predictions. Setting eval_metric="AUC" causes cross-fold calibration
# issues because AUC-based early stopping picks different iterations
# per fold with inconsistent probability scales.
HISTORICAL_PARAMS = {
    "iterations": 1183,
    "learning_rate": 0.010492928203031996,
    "depth": 4,
    "l2_leaf_reg": 4.5594175959311425,
    "min_data_in_leaf": 7,
    "verbose": 0,
    "allow_writing_files": False,
}


# ===================================================================
# Feature matrices — exact replicas of what worked
# ===================================================================
def build_original_36_features(train_df, test_df):
    """EXACT copy of the 36-feature matrix from the 0.844 LB submission."""
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

    for col in CAT_COLS:
        df[col] = df[col].fillna("Unknown").astype("category")

    X_train = df.iloc[: len(train_df)].drop(columns=["Id"]).reset_index(drop=True)
    X_test = df.iloc[len(train_df) :].drop(columns=["Id"]).reset_index(drop=True)
    return X_train, X_test


def build_36_plus_safe(train_df, test_df):
    """36 original features + a few very safe additions."""
    df = pd.concat([train_df.drop(columns=["Drafted"]), test_df], axis=0).reset_index(drop=True)

    # ── Original missingness flags ──
    for col in PHYSICAL:
        if df[col].isnull().sum() > 0:
            df[f"{col}_Is_Missing"] = df[col].isnull().astype(int)

    # ── SAFE NEW: aggregate missingness (very robust, no position interaction) ──
    combine_drills = ["Sprint_40yd", "Vertical_Jump", "Bench_Press_Reps",
                      "Broad_Jump", "Agility_3cone", "Shuttle"]
    df["n_combine_missing"] = df[combine_drills].isnull().sum(axis=1)

    # ── Original imputation ──
    for col in PHYSICAL:
        df[col] = df[col].fillna(df.groupby("Position")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

    # ── Original derived features ──
    df["BMI"] = df["Weight"] / (df["Height"] ** 2)
    df["Speed_Weight_Ratio"] = df["Weight"] / df["Sprint_40yd"]
    df["Explosiveness_Index"] = df["Vertical_Jump"] * df["Broad_Jump"]
    df["Power_Score"] = df["Bench_Press_Reps"] * df["Weight"]
    df["School_Combine_Count"] = df.groupby("School")["Id"].transform("count")
    df["Speed_Score"] = (df["Weight"] * 200) / (df["Sprint_40yd"] ** 4)

    # ── Original position-relative diffs ──
    for col in PHYSICAL:
        df[f"{col}_Pos_Mean"] = df.groupby("Position")[col].transform("mean")
        df[f"{col}_Pos_Diff"] = df[col] - df[f"{col}_Pos_Mean"]
        df.drop(columns=[f"{col}_Pos_Mean"], inplace=True)

    # ── SAFE NEW: burst score and agility total (simple, robust) ──
    df["Burst_Score"] = df["Vertical_Jump"] + df["Broad_Jump"]
    df["Agility_Total"] = df["Agility_3cone"] + df["Shuttle"]

    # ── SAFE NEW: position-level percentile for the 3 most correlated metrics ──
    for col in ["Sprint_40yd", "Bench_Press_Reps", "Age"]:
        df[f"{col}_Pos_Pctile"] = df.groupby("Position")[col].rank(pct=True, method="average")

    for col in CAT_COLS:
        df[col] = df[col].fillna("Unknown").astype("category")

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train = df.iloc[: len(train_df)].drop(columns=["Id"]).reset_index(drop=True)
    X_test = df.iloc[len(train_df) :].drop(columns=["Id"]).reset_index(drop=True)
    return X_train, X_test


# ===================================================================
# Multi-seed 10-fold CatBoost trainer
# ===================================================================
def train_catboost_multiseed(
    X, X_test, y, params, name, n_seeds=5, n_folds=10,
):
    """Train CatBoost with multiple seeds and average predictions."""
    cat_indices = [i for i, c in enumerate(X.columns) if c in CAT_COLS]

    all_oof = []
    all_test = []

    for seed in range(n_seeds):
        base_seed = 42 + seed * 1000
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=base_seed)
        oof = np.zeros(len(X))
        test_pred = np.zeros(len(X_test))

        for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
            p = params.copy()
            p["random_seed"] = base_seed + fold

            model = CatBoostClassifier(**p, cat_features=cat_indices)
            model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=(X.iloc[va_idx], y.iloc[va_idx]),
                early_stopping_rounds=50,
                use_best_model=True,
            )
            oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
            test_pred += model.predict_proba(X_test)[:, 1] / n_folds

        seed_auc = roc_auc_score(y, oof)
        print(f"    {name} seed {seed}: OOF = {seed_auc:.6f}", flush=True)
        all_oof.append(oof)
        all_test.append(test_pred)

    avg_oof = np.mean(all_oof, axis=0)
    avg_test = np.mean(all_test, axis=0)
    auc = roc_auc_score(y, avg_oof)
    print(f"  >> {name} multi-seed OOF: {auc:.6f}", flush=True)
    return {"name": name, "oof": avg_oof, "test_pred": avg_test, "auc": auc}


# ===================================================================
# Single-seed 10-fold (exact replica of old submission)
# ===================================================================
def train_catboost_single(X, X_test, y, params, name, n_folds=10, seed=42):
    """Exact replica of the old 0.844 submission approach."""
    cat_indices = [i for i, c in enumerate(X.columns) if c in CAT_COLS]
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        p = params.copy()
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
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(score)
        print(f"    {name} fold {fold:02d}: {score:.5f}", flush=True)

    auc = roc_auc_score(y, oof)
    print(f"  >> {name} OOF: {auc:.6f} (mean fold: {np.mean(fold_scores):.6f})", flush=True)
    return {"name": name, "oof": oof, "test_pred": test_pred, "auc": auc}


def rank01(v):
    return pd.Series(v).rank(pct=True, method="average").values


# ===================================================================
# Main
# ===================================================================
def main():
    t0 = time.time()

    print("=" * 60)
    print("ANTIGRAVITY v3 - Conservative, LB-Proven Approach")
    print("=" * 60)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    y = train["Drafted"].reset_index(drop=True)
    test_ids = test["Id"].reset_index(drop=True)
    print(f"Train: {train.shape} | Test: {test.shape}")

    # ── Build feature matrices ──────────────────────────────────────
    print("\n[1] Building feature matrices...")
    X36, X36_test = build_original_36_features(train, test)
    X36p, X36p_test = build_36_plus_safe(train, test)
    print(f"  Original 36 features: {X36.shape[1]} columns")
    print(f"  36 + safe additions: {X36p.shape[1]} columns")

    candidates = []

    # ── Candidate 1: EXACT replica of old 0.844 submission ─────────
    print("\n[2] Exact replica of old 0.844 submission (single seed, 10-fold)...")
    candidates.append(train_catboost_single(
        X36, X36_test, y, HISTORICAL_PARAMS,
        "exact_replica_old", n_folds=10, seed=42,
    ))

    # ── Candidate 2: Old features, multi-seed (reduce variance) ────
    print("\n[3] Old 36 features + multi-seed (5 seeds x 10 folds)...")
    candidates.append(train_catboost_multiseed(
        X36, X36_test, y, HISTORICAL_PARAMS,
        "old_36f_multiseed", n_seeds=5,
    ))

    # ── Candidate 3: Old features, multi-seed, MORE seeds ──────────
    print("\n[4] Old 36 features + multi-seed (10 seeds x 10 folds)...")
    candidates.append(train_catboost_multiseed(
        X36, X36_test, y, HISTORICAL_PARAMS,
        "old_36f_10seeds", n_seeds=10,
    ))

    # ── Candidate 4: 36+safe features, multi-seed ─────────────────
    print("\n[5] 36+safe features + multi-seed (5 seeds)...")
    candidates.append(train_catboost_multiseed(
        X36p, X36p_test, y, HISTORICAL_PARAMS,
        "safe_features_multiseed", n_seeds=5,
    ))

    # ── Candidate 5: Old features, slightly different params ────────
    # More regularized version (higher l2, fewer iterations)
    print("\n[6] Old 36 features, more regularized params, multi-seed...")
    reg_params = HISTORICAL_PARAMS.copy()
    reg_params["l2_leaf_reg"] = 8.0
    reg_params["min_data_in_leaf"] = 12
    reg_params["depth"] = 4
    candidates.append(train_catboost_multiseed(
        X36, X36_test, y, reg_params,
        "old_36f_regularized", n_seeds=5,
    ))

    # ── Candidate 6: Old features, depth=3 (simpler trees) ─────────
    print("\n[7] Old 36 features, depth=3 (simpler), multi-seed...")
    d3_params = HISTORICAL_PARAMS.copy()
    d3_params["depth"] = 3
    d3_params["iterations"] = 1500
    candidates.append(train_catboost_multiseed(
        X36, X36_test, y, d3_params,
        "old_36f_depth3", n_seeds=5,
    ))

    # ── Simple equal-weight blends (NO fitted weights) ─────────────
    print("\n[8] Building conservative blends (equal weight only)...")
    blends = []

    # Blend top-2 by OOF (rank-based, equal weight)
    aucs = [c["auc"] for c in candidates]
    top2_idx = np.argsort(aucs)[-2:]
    top2_oof_rank = np.column_stack([rank01(candidates[i]["oof"]) for i in top2_idx])
    top2_test_rank = np.column_stack([rank01(candidates[i]["test_pred"]) for i in top2_idx])
    blend_oof = top2_oof_rank.mean(axis=1)
    blend_test = top2_test_rank.mean(axis=1)
    blend_auc = roc_auc_score(y, blend_oof)
    top2_names = [candidates[i]["name"] for i in top2_idx]
    blends.append({"name": f"blend_top2_equal", "oof": blend_oof, "test_pred": blend_test, "auc": blend_auc})
    print(f"  >> blend_top2_equal ({top2_names}) OOF: {blend_auc:.6f}", flush=True)

    # Blend top-3
    top3_idx = np.argsort(aucs)[-3:]
    top3_oof_rank = np.column_stack([rank01(candidates[i]["oof"]) for i in top3_idx])
    top3_test_rank = np.column_stack([rank01(candidates[i]["test_pred"]) for i in top3_idx])
    blend_oof3 = top3_oof_rank.mean(axis=1)
    blend_test3 = top3_test_rank.mean(axis=1)
    blend_auc3 = roc_auc_score(y, blend_oof3)
    top3_names = [candidates[i]["name"] for i in top3_idx]
    blends.append({"name": f"blend_top3_equal", "oof": blend_oof3, "test_pred": blend_test3, "auc": blend_auc3})
    print(f"  >> blend_top3_equal ({top3_names}) OOF: {blend_auc3:.6f}", flush=True)

    # Blend: old replica + multi-seed (prob average, not rank)
    prob_blend_oof = (candidates[0]["oof"] + candidates[1]["oof"]) / 2
    prob_blend_test = (candidates[0]["test_pred"] + candidates[1]["test_pred"]) / 2
    prob_blend_auc = roc_auc_score(y, prob_blend_oof)
    blends.append({"name": "blend_replica_plus_ms", "oof": prob_blend_oof, "test_pred": prob_blend_test, "auc": prob_blend_auc})
    print(f"  >> blend_replica_plus_ms OOF: {prob_blend_auc:.6f}", flush=True)

    # ── Save all ───────────────────────────────────────────────────
    print("\n[9] Saving submissions...")
    all_results = candidates + blends
    rows = []
    for r in all_results:
        path = SUB_DIR / f"{r['name']}.csv"
        pd.DataFrame({"Id": test_ids, "Drafted": r["test_pred"]}).to_csv(path, index=False)
        rows.append({"name": r["name"], "oof_auc": r["auc"], "file": path.name})

    summary = pd.DataFrame(rows).sort_values("oof_auc", ascending=False)

    # Best
    best = all_results[int(np.argmax([r["auc"] for r in all_results]))]
    best_path = SUB_DIR / "BEST_V3.csv"
    pd.DataFrame({"Id": test_ids, "Drafted": best["test_pred"]}).to_csv(best_path, index=False)

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("V3 FINAL RESULTS")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nBEST: {best['name']} (OOF AUC: {best['auc']:.6f})")
    print(f"Saved to: {best_path}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    print("\nSUBMIT ORDER (safest first):")
    print("  1. old_36f_multiseed - same proven features, just less variance")
    print("  2. old_36f_10seeds - even more variance reduction")
    print("  3. exact_replica_old - sanity check (should match 0.844)")
    print("  4. blend_replica_plus_ms - if multi-seed helps, this averages both")
    print("  5. safe_features_multiseed - test if extra features help or hurt")


if __name__ == "__main__":
    main()

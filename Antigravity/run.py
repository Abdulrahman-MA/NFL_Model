"""
Antigravity — Main Runner for NFL Draft Legal-Limit Push.

Orchestrates: features → tune → train → stack → submit.
Usage: python Antigravity/run.py [--tune] [--trials N] [--force-retune]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add Antigravity to path so imports work
sys.path.insert(0, str(Path(__file__).parent))

from features import build_features, CAT_COLS
from tune_models import tune_all
from stack_and_submit import (
    train_catboost, train_lgbm, train_xgb, train_extratrees,
    build_blends_and_stack, save_all_submissions,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
RESULTS_DIR = Path(__file__).parent / "results"


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    y = train["Drafted"].reset_index(drop=True)
    test_ids = test["Id"].reset_index(drop=True)
    return train, test, y, test_ids


def run_pipeline(do_tune: bool = True, n_trials: int = 80, force_retune: bool = False):
    t0 = time.time()

    # ── 1. Load data ────────────────────────────────────────────────────
    print("=" * 60)
    print("ANTIGRAVITY — NFL Draft Legal-Limit Pipeline")
    print("=" * 60)
    train, test, y, test_ids = load_data()
    print(f"Train: {train.shape} | Test: {test.shape}")
    print(f"Target: {int(y.sum())} drafted / {int((1-y).sum())} undrafted ({y.mean():.3f})")

    # ── 2. Build features ───────────────────────────────────────────────
    print("\n[Phase 1] Building advanced features...")
    X_train, X_test = build_features(train, test)
    print(f"  Feature matrix: {X_train.shape[1]} columns")
    print(f"  Categorical: {[c for c in CAT_COLS if c in X_train.columns]}")
    print(f"  Numeric: {X_train.shape[1] - sum(1 for c in CAT_COLS if c in X_train.columns)}")

    # ── 3. Tune hyperparameters ─────────────────────────────────────────
    if do_tune:
        print(f"\n[Phase 2] Optuna hyperparameter tuning ({n_trials} trials per model)...")
        tuned = tune_all(X_train, X_test, y, n_trials=n_trials, force_retune=force_retune)
    else:
        print("\n[Phase 2] Skipping tuning, using default parameters...")
        tuned = _default_params()

    # ── 4. Train all models (10-fold CV) ────────────────────────────────
    print("\n[Phase 3] Training all models with 10-fold CV...")
    candidates = []

    # CatBoost with tuned params
    if "catboost" in tuned:
        cb_params = tuned["catboost"]["best_params"].copy()
        print(f"\n  Training CatBoost (tuned)...")
        candidates.append(train_catboost(X_train, X_test, y, cb_params, name="catboost_tuned"))

    # Also train CatBoost with the known-good historical params (0.844 submission)
    print(f"\n  Training CatBoost (historical best)...")
    hist_params = {
        "iterations": 1183,
        "learning_rate": 0.010492928203031996,
        "depth": 4,
        "l2_leaf_reg": 4.5594175959311425,
        "min_data_in_leaf": 7,
    }
    candidates.append(train_catboost(X_train, X_test, y, hist_params, name="catboost_historical"))

    # LightGBM
    if "lightgbm" in tuned:
        lgbm_params = tuned["lightgbm"]["best_params"].copy()
        print(f"\n  Training LightGBM (tuned)...")
        candidates.append(train_lgbm(X_train, X_test, y, lgbm_params, name="lgbm_tuned"))

    # XGBoost
    if "xgboost" in tuned:
        xgb_params = tuned["xgboost"]["best_params"].copy()
        print(f"\n  Training XGBoost (tuned)...")
        candidates.append(train_xgb(X_train, X_test, y, xgb_params, name="xgb_tuned"))

    # ExtraTrees
    if "extratrees" in tuned:
        et_params = tuned["extratrees"]["best_params"].copy()
        print(f"\n  Training ExtraTrees (tuned)...")
        candidates.append(train_extratrees(X_train, X_test, y, et_params, name="et_tuned"))

    # ── 5. Stacking & blending ──────────────────────────────────────────
    print("\n[Phase 4] Building blends and stacking...")
    blends = build_blends_and_stack(candidates, y, test_ids)

    # ── 6. Save & report ────────────────────────────────────────────────
    print("\n[Phase 5] Saving submissions and generating report...")
    summary = save_all_submissions(candidates, blends, test_ids)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("FINAL RESULTS TABLE")
    print("=" * 60)
    print(summary.to_string(index=False))

    elapsed = time.time() - t0
    print(f"\nTotal pipeline time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Save submit order recommendation
    submit_order = summary.head(5)["name"].tolist()
    print("\n📋 RECOMMENDED SUBMIT ORDER:")
    for i, name in enumerate(submit_order, 1):
        row = summary[summary["name"] == name].iloc[0]
        print(f"  {i}. {name} (OOF: {row['oof_auc']:.6f}) → Antigravity/submissions/{name}.csv")

    return summary


def _default_params() -> dict:
    """Fallback params if tuning is skipped."""
    return {
        "catboost": {
            "best_params": {
                "iterations": 1500,
                "learning_rate": 0.015,
                "depth": 5,
                "l2_leaf_reg": 5.0,
                "min_data_in_leaf": 10,
                "subsample": 0.8,
                "colsample_bylevel": 0.7,
                "random_strength": 1.0,
                "bagging_temperature": 1.0,
                "bootstrap_type": "Bayesian",
            },
            "best_value": 0.0,
        },
        "lightgbm": {
            "best_params": {
                "n_estimators": 1800,
                "learning_rate": 0.012,
                "num_leaves": 15,
                "max_depth": 4,
                "min_child_samples": 35,
                "subsample": 0.82,
                "colsample_bytree": 0.70,
                "reg_alpha": 0.2,
                "reg_lambda": 4.0,
                "min_split_gain": 0.1,
                "feature_fraction_bynode": 0.7,
            },
            "best_value": 0.0,
        },
        "xgboost": {
            "best_params": {
                "n_estimators": 1400,
                "learning_rate": 0.012,
                "max_depth": 3,
                "min_child_weight": 8,
                "subsample": 0.82,
                "colsample_bytree": 0.72,
                "reg_alpha": 0.1,
                "reg_lambda": 5.0,
                "gamma": 0.1,
            },
            "best_value": 0.0,
        },
        "extratrees": {
            "best_params": {
                "n_estimators": 900,
                "max_depth": 15,
                "min_samples_leaf": 4,
                "min_samples_split": 5,
                "max_features": 0.5,
                "class_weight": "none",
            },
            "best_value": 0.0,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Antigravity NFL Draft Pipeline")
    parser.add_argument("--tune", action="store_true", default=True,
                        help="Run Optuna tuning (default: True)")
    parser.add_argument("--no-tune", dest="tune", action="store_false",
                        help="Skip Optuna tuning, use default params")
    parser.add_argument("--trials", type=int, default=80,
                        help="Number of Optuna trials per model (default: 80)")
    parser.add_argument("--force-retune", action="store_true",
                        help="Force retuning even if cached params exist")
    args = parser.parse_args()

    run_pipeline(do_tune=args.tune, n_trials=args.trials, force_retune=args.force_retune)

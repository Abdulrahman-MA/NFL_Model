"""
Antigravity v2 — Diagnostic + Improved Pipeline.

Key fixes from v1:
1. The OOF-vs-fold-mean gap (0.823 vs 0.847 for CatBoost) was caused by 
   target encoding creating inconsistent feature spaces across folds.
   Fix: CatBoost uses native categoricals only (no target encoding).
   Fix: For LGBM/XGB, use a simpler more robust target encoding.
2. 237 features with 2781 samples → overfitting. 
   Fix: Feature importance-based selection to keep top ~60-80 features.
3. Multi-seed ensembling to reduce variance.
4. Focus on what WORKS: ExtraTrees had the best single-model OOF (0.839)
   and CatBoost had the best fold_mean (0.847). Lean into both.

Usage: .venv\\Scripts\\python.exe Antigravity/run_v2.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features, CAT_COLS, add_target_encoding_fold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
SUB_DIR = Path(__file__).parent / "submissions_v2"
RESULTS_DIR = Path(__file__).parent / "results"
PARAMS_DIR = Path(__file__).parent / "tuned_params"

N_FOLDS = 10
N_SEEDS = 5  # Multi-seed ensembling


def rank01(v):
    return pd.Series(v).rank(pct=True, method="average").values


def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    y = train["Drafted"].reset_index(drop=True)
    test_ids = test["Id"].reset_index(drop=True)
    return train, test, y, test_ids


def load_tuned_params():
    """Load cached Optuna params from v1 run."""
    params = {}
    for name in ["catboost", "lightgbm", "xgboost", "extratrees"]:
        path = PARAMS_DIR / f"{name}_best.json"
        if path.exists():
            with open(path) as f:
                params[name] = json.load(f)
    return params


# =======================================================================
# Feature selection via importance
# =======================================================================
def select_features_by_importance(
    X_train: pd.DataFrame, y: pd.Series, top_n: int = 80,
) -> list[str]:
    """Use CatBoost feature importance to select top features."""
    print(f"  Running feature importance selection (keeping top {top_n})...")
    
    cat_indices = [i for i, c in enumerate(X_train.columns) if c in CAT_COLS]
    X_cb = X_train.copy()
    for col in CAT_COLS:
        if col in X_cb.columns:
            X_cb[col] = X_cb[col].astype("category")
    
    model = CatBoostClassifier(
        iterations=800, learning_rate=0.03, depth=5,
        l2_leaf_reg=10, min_data_in_leaf=20,
        loss_function="Logloss", eval_metric="AUC",
        verbose=0, allow_writing_files=False,
        random_seed=42, cat_features=cat_indices,
    )
    model.fit(X_cb, y)
    
    importances = model.get_feature_importance()
    feat_imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    
    # Always keep categorical columns
    selected = list(feat_imp.head(top_n)["feature"])
    for cat in CAT_COLS:
        if cat not in selected and cat in X_train.columns:
            selected.append(cat)
    
    print(f"  Selected {len(selected)} features")
    print(f"  Top 15: {list(feat_imp.head(15)['feature'])}")
    return selected


# =======================================================================
# Multi-seed CatBoost (native categoricals, no target encoding)
# =======================================================================
def train_catboost_multiseed(
    X: pd.DataFrame, X_test: pd.DataFrame, y: pd.Series,
    params: dict, name: str = "catboost_ms",
) -> dict:
    """Multi-seed 10-fold CatBoost. Native categoricals only."""
    cat_indices = [i for i, c in enumerate(X.columns) if c in CAT_COLS]
    
    all_oof = []
    all_test = []
    
    for seed in range(N_SEEDS):
        base_seed = 42 + seed * 1000
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=base_seed)
        oof = np.zeros(len(X))
        test_pred = np.zeros(len(X_test))
        
        X_cb = X.copy()
        X_test_cb = X_test.copy()
        for col in CAT_COLS:
            if col in X_cb.columns:
                X_cb[col] = X_cb[col].astype("category")
                X_test_cb[col] = X_test_cb[col].astype("category")
        
        for fold, (tr_idx, va_idx) in enumerate(cv.split(X_cb, y), 1):
            p = params.copy()
            p.pop("class_weight", None)
            p["random_seed"] = base_seed + fold
            p["loss_function"] = "Logloss"
            p["eval_metric"] = "AUC"
            p["verbose"] = 0
            p["allow_writing_files"] = False
            p["thread_count"] = -1
            if "bootstrap_type" not in p:
                p["bootstrap_type"] = "Bayesian"
            if p.get("bootstrap_type") == "Bayesian":
                p.pop("subsample", None)
            
            model = CatBoostClassifier(**p, cat_features=cat_indices)
            model.fit(
                X_cb.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=(X_cb.iloc[va_idx], y.iloc[va_idx]),
                early_stopping_rounds=80,
                use_best_model=True,
            )
            oof[va_idx] = model.predict_proba(X_cb.iloc[va_idx])[:, 1]
            test_pred += model.predict_proba(X_test_cb)[:, 1] / N_FOLDS
        
        seed_auc = roc_auc_score(y, oof)
        print(f"    {name} seed {seed}: OOF AUC = {seed_auc:.6f}", flush=True)
        all_oof.append(oof)
        all_test.append(test_pred)
    
    # Average across seeds
    avg_oof = np.mean(all_oof, axis=0)
    avg_test = np.mean(all_test, axis=0)
    auc = roc_auc_score(y, avg_oof)
    print(f"  >> {name} multi-seed OOF AUC: {auc:.6f}", flush=True)
    return {"name": name, "oof": avg_oof, "test_pred": avg_test, "auc": auc, "fold_scores": []}


# =======================================================================
# Multi-seed XGBoost (with target encoding)
# =======================================================================
def train_xgb_multiseed(
    X: pd.DataFrame, X_test: pd.DataFrame, y: pd.Series,
    params: dict, name: str = "xgb_ms",
) -> dict:
    """Multi-seed 10-fold XGBoost with fold-level target encoding."""
    all_oof = []
    all_test = []
    
    for seed in range(N_SEEDS):
        base_seed = 42 + seed * 1000
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=base_seed)
        oof = np.zeros(len(X))
        test_pred = np.zeros(len(X_test))
        
        for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
            X_tr, X_va, X_te = add_target_encoding_fold(X, y, X_test, tr_idx, va_idx)
            drop_cols = [c for c in CAT_COLS if c in X_tr.columns]
            X_tr = X_tr.drop(columns=drop_cols).replace([np.inf, -np.inf], np.nan).fillna(0)
            X_va = X_va.drop(columns=drop_cols).replace([np.inf, -np.inf], np.nan).fillna(0)
            X_te = X_te.drop(columns=drop_cols).replace([np.inf, -np.inf], np.nan).fillna(0)
            
            p = params.copy()
            p.pop("class_weight", None)
            p["objective"] = "binary:logistic"
            p["eval_metric"] = "auc"
            p["tree_method"] = "hist"
            p["verbosity"] = 0
            es = p.pop("early_stopping_rounds", 80)
            p["early_stopping_rounds"] = es
            
            model = XGBClassifier(**p, random_state=base_seed + fold, n_jobs=-1)
            model.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            test_pred += model.predict_proba(X_te)[:, 1] / N_FOLDS
        
        seed_auc = roc_auc_score(y, oof)
        print(f"    {name} seed {seed}: OOF AUC = {seed_auc:.6f}", flush=True)
        all_oof.append(oof)
        all_test.append(test_pred)
    
    avg_oof = np.mean(all_oof, axis=0)
    avg_test = np.mean(all_test, axis=0)
    auc = roc_auc_score(y, avg_oof)
    print(f"  >> {name} multi-seed OOF AUC: {auc:.6f}", flush=True)
    return {"name": name, "oof": avg_oof, "test_pred": avg_test, "auc": auc, "fold_scores": []}


# =======================================================================
# Multi-seed ExtraTrees
# =======================================================================
def train_et_multiseed(
    X: pd.DataFrame, X_test: pd.DataFrame, y: pd.Series,
    params: dict, name: str = "et_ms",
) -> dict:
    """Multi-seed 10-fold ExtraTrees with fold-level target encoding."""
    all_oof = []
    all_test = []
    
    for seed in range(N_SEEDS):
        base_seed = 42 + seed * 1000
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=base_seed)
        oof = np.zeros(len(X))
        test_pred = np.zeros(len(X_test))
        
        for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
            X_tr, X_va, X_te = add_target_encoding_fold(X, y, X_test, tr_idx, va_idx)
            drop_cols = [c for c in CAT_COLS if c in X_tr.columns]
            X_tr = X_tr.drop(columns=drop_cols).replace([np.inf, -np.inf], np.nan).fillna(0)
            X_va = X_va.drop(columns=drop_cols).replace([np.inf, -np.inf], np.nan).fillna(0)
            X_te = X_te.drop(columns=drop_cols).replace([np.inf, -np.inf], np.nan).fillna(0)
            
            p = params.copy()
            cw = p.pop("class_weight", None)
            if cw == "none" or cw is None:
                p["class_weight"] = None
            else:
                p["class_weight"] = cw
            p["n_jobs"] = -1
            
            model = ExtraTreesClassifier(**p, random_state=base_seed + fold)
            model.fit(X_tr, y.iloc[tr_idx])
            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            test_pred += model.predict_proba(X_te)[:, 1] / N_FOLDS
        
        seed_auc = roc_auc_score(y, oof)
        print(f"    {name} seed {seed}: OOF AUC = {seed_auc:.6f}", flush=True)
        all_oof.append(oof)
        all_test.append(test_pred)
    
    avg_oof = np.mean(all_oof, axis=0)
    avg_test = np.mean(all_test, axis=0)
    auc = roc_auc_score(y, avg_oof)
    print(f"  >> {name} multi-seed OOF AUC: {auc:.6f}", flush=True)
    return {"name": name, "oof": avg_oof, "test_pred": avg_test, "auc": auc, "fold_scores": []}


# =======================================================================
# Multi-seed LightGBM
# =======================================================================
def train_lgbm_multiseed(
    X: pd.DataFrame, X_test: pd.DataFrame, y: pd.Series,
    params: dict, name: str = "lgbm_ms",
) -> dict:
    """Multi-seed 10-fold LightGBM with fold-level target encoding."""
    all_oof = []
    all_test = []
    
    for seed in range(N_SEEDS):
        base_seed = 42 + seed * 1000
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=base_seed)
        oof = np.zeros(len(X))
        test_pred = np.zeros(len(X_test))
        
        for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
            X_tr, X_va, X_te = add_target_encoding_fold(X, y, X_test, tr_idx, va_idx)
            drop_cols = [c for c in CAT_COLS if c in X_tr.columns]
            X_tr = X_tr.drop(columns=drop_cols).replace([np.inf, -np.inf], np.nan).fillna(0)
            X_va = X_va.drop(columns=drop_cols).replace([np.inf, -np.inf], np.nan).fillna(0)
            X_te = X_te.drop(columns=drop_cols).replace([np.inf, -np.inf], np.nan).fillna(0)
            
            p = params.copy()
            p.pop("class_weight", None)
            p["objective"] = "binary"
            p["metric"] = "auc"
            p["verbose"] = -1
            
            model = LGBMClassifier(**p, random_state=base_seed + fold, n_jobs=-1)
            model.fit(
                X_tr, y.iloc[tr_idx],
                eval_set=[(X_va, y.iloc[va_idx])],
                callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
            )
            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            test_pred += model.predict_proba(X_te)[:, 1] / N_FOLDS
        
        seed_auc = roc_auc_score(y, oof)
        print(f"    {name} seed {seed}: OOF AUC = {seed_auc:.6f}", flush=True)
        all_oof.append(oof)
        all_test.append(test_pred)
    
    avg_oof = np.mean(all_oof, axis=0)
    avg_test = np.mean(all_test, axis=0)
    auc = roc_auc_score(y, avg_oof)
    print(f"  >> {name} multi-seed OOF AUC: {auc:.6f}", flush=True)
    return {"name": name, "oof": avg_oof, "test_pred": avg_test, "auc": auc, "fold_scores": []}


# =======================================================================
# Blending
# =======================================================================
def build_blends(candidates, y, test_ids):
    blends = []
    oof_prob = np.column_stack([c["oof"] for c in candidates])
    test_prob = np.column_stack([c["test_pred"] for c in candidates])
    oof_rank = np.column_stack([rank01(c["oof"]) for c in candidates])
    test_rank = np.column_stack([rank01(c["test_pred"]) for c in candidates])
    
    def _add(name, oof_pred, test_pred):
        auc = roc_auc_score(y, oof_pred)
        blends.append({"name": name, "oof": oof_pred, "test_pred": test_pred, "auc": auc, "fold_scores": []})
        print(f"  >> {name} OOF AUC: {auc:.6f}", flush=True)
    
    # Simple blends
    _add("v2_blend_prob_mean", oof_prob.mean(axis=1), test_prob.mean(axis=1))
    _add("v2_blend_rank_mean", oof_rank.mean(axis=1), test_rank.mean(axis=1))
    
    # Top-2 and Top-3
    aucs = [c["auc"] for c in candidates]
    top2 = np.argsort(aucs)[-2:]
    top3 = np.argsort(aucs)[-3:]
    _add("v2_blend_rank_top2", oof_rank[:, top2].mean(axis=1), test_rank[:, top2].mean(axis=1))
    _add("v2_blend_rank_top3", oof_rank[:, top3].mean(axis=1), test_rank[:, top3].mean(axis=1))
    
    # Conservative weighted blend (only 2000 trials to avoid OOF overfit)
    rng = np.random.default_rng(42)
    best_auc = -1.0
    best_w = None
    for _ in range(2000):
        w = rng.dirichlet(np.ones(len(candidates)))
        auc = roc_auc_score(y, oof_rank @ w)
        if auc > best_auc:
            best_auc = auc
            best_w = w
    _add("v2_blend_rank_weighted", oof_rank @ best_w, test_rank @ best_w)
    print(f"    Weights: {dict(zip([c['name'] for c in candidates], best_w.round(4)))}")
    
    # LR meta-stack
    meta_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2026)
    meta_oof = np.zeros(len(y))
    for tr_idx, va_idx in meta_cv.split(oof_prob, y):
        meta = LogisticRegressionCV(
            cv=5, scoring="roc_auc", max_iter=5000, Cs=20,
            penalty="l2", n_jobs=-1, random_state=42,
        )
        meta.fit(oof_prob[tr_idx], y.iloc[tr_idx])
        meta_oof[va_idx] = meta.predict_proba(oof_prob[va_idx])[:, 1]
    final_meta = LogisticRegressionCV(
        cv=10, scoring="roc_auc", max_iter=5000, Cs=20,
        penalty="l2", n_jobs=-1, random_state=42,
    )
    final_meta.fit(oof_prob, y)
    meta_test = final_meta.predict_proba(test_prob)[:, 1]
    _add("v2_stack_lr", meta_oof, meta_test)
    
    return blends


# =======================================================================
# Main
# =======================================================================
def main():
    t0 = time.time()
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ANTIGRAVITY v2 - Multi-Seed + Feature Selection Pipeline")
    print("=" * 60)
    
    # Load data
    train, test, y, test_ids = load_data()
    print(f"Train: {train.shape} | Test: {test.shape}")
    
    # Build features
    print("\n[1] Building features...")
    X_train_full, X_test_full = build_features(train, test)
    print(f"  Full feature matrix: {X_train_full.shape[1]} columns")
    
    # Feature selection
    print("\n[2] Feature selection...")
    selected = select_features_by_importance(X_train_full, y, top_n=80)
    X_train = X_train_full[selected].copy()
    X_test = X_test_full[selected].copy()
    print(f"  Reduced to {X_train.shape[1]} features")
    
    # Load tuned params from v1
    print("\n[3] Loading Optuna-tuned params from v1...")
    tuned = load_tuned_params()
    for name, p in tuned.items():
        print(f"  {name}: best_value = {p['best_value']:.6f}")
    
    # Train all models with multi-seed
    print(f"\n[4] Training models ({N_SEEDS} seeds x {N_FOLDS} folds each)...")
    candidates = []
    
    # CatBoost (Optuna-tuned params)
    print(f"\n  --- CatBoost (tuned, multi-seed) ---")
    cb_params = tuned["catboost"]["best_params"].copy()
    candidates.append(train_catboost_multiseed(X_train, X_test, y, cb_params, "cb_tuned_ms"))
    
    # CatBoost (historical params that scored 0.844 on LB)
    print(f"\n  --- CatBoost (historical, multi-seed) ---")
    hist_params = {
        "iterations": 1183,
        "learning_rate": 0.010492928203031996,
        "depth": 4,
        "l2_leaf_reg": 4.5594175959311425,
        "min_data_in_leaf": 7,
    }
    candidates.append(train_catboost_multiseed(X_train, X_test, y, hist_params, "cb_hist_ms"))
    
    # XGBoost (tuned)
    print(f"\n  --- XGBoost (tuned, multi-seed) ---")
    xgb_params = tuned["xgboost"]["best_params"].copy()
    candidates.append(train_xgb_multiseed(X_train, X_test, y, xgb_params, "xgb_tuned_ms"))
    
    # ExtraTrees (tuned)
    print(f"\n  --- ExtraTrees (tuned, multi-seed) ---")
    et_params = tuned["extratrees"]["best_params"].copy()
    candidates.append(train_et_multiseed(X_train, X_test, y, et_params, "et_tuned_ms"))
    
    # LightGBM (tuned)
    print(f"\n  --- LightGBM (tuned, multi-seed) ---")
    lgbm_params = tuned["lightgbm"]["best_params"].copy()
    candidates.append(train_lgbm_multiseed(X_train, X_test, y, lgbm_params, "lgbm_tuned_ms"))
    
    # Blending
    print(f"\n[5] Building blends and stacking...")
    blends = build_blends(candidates, y, test_ids)
    
    # Save everything
    print(f"\n[6] Saving submissions...")
    all_results = candidates + blends
    rows = []
    for r in all_results:
        path = SUB_DIR / f"{r['name']}.csv"
        pd.DataFrame({"Id": test_ids, "Drafted": r["test_pred"]}).to_csv(path, index=False)
        rows.append({"name": r["name"], "oof_auc": r["auc"], "file": path.name})
    
    summary = pd.DataFrame(rows).sort_values("oof_auc", ascending=False)
    summary.to_csv(RESULTS_DIR / "v2_summary.csv", index=False)
    
    # Best submission
    best = all_results[int(np.argmax([r["auc"] for r in all_results]))]
    best_path = SUB_DIR / "BEST_V2.csv"
    pd.DataFrame({"Id": test_ids, "Drafted": best["test_pred"]}).to_csv(best_path, index=False)
    
    elapsed = time.time() - t0
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nBEST: {best['name']} (OOF AUC: {best['auc']:.6f})")
    print(f"Saved to: {best_path}")
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    
    print("\nRECOMMENDED SUBMIT ORDER:")
    for i, row in summary.head(5).iterrows():
        print(f"  {row['name']} (OOF: {row['oof_auc']:.6f}) -> Antigravity/submissions_v2/{row['file']}")


if __name__ == "__main__":
    main()

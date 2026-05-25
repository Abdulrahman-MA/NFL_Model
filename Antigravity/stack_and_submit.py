"""
Antigravity — Stacking, Blending & Submission Generation.

Takes tuned hyperparameters and the feature matrix, trains all models
with 10-fold CV, builds a 2-level stack, and generates candidate submissions.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import CAT_COLS, add_target_encoding_fold

warnings.filterwarnings("ignore")

SUB_DIR = Path(__file__).parent / "submissions"
RESULTS_DIR = Path(__file__).parent / "results"

N_FOLDS = 10
SEED = 42


def _rank01(v: np.ndarray) -> np.ndarray:
    return pd.Series(v).rank(pct=True, method="average").values


# ===========================================================================
# Individual model trainers (10-fold CV)
# ===========================================================================
def train_catboost(
    X: pd.DataFrame, X_test: pd.DataFrame, y: pd.Series,
    params: dict, name: str = "catboost",
) -> dict:
    """10-fold CatBoost with native categoricals."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores = []

    cat_indices = [i for i, c in enumerate(X.columns) if c in CAT_COLS]

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        X_te = X_test.copy()

        for col in CAT_COLS:
            if col in X_tr.columns:
                X_tr[col] = X_tr[col].astype("category")
                X_va[col] = X_va[col].astype("category")
                X_te[col] = X_te[col].astype("category")

        p = params.copy()
        p.pop("class_weight", None)
        p["random_seed"] = SEED + fold
        p["loss_function"] = "Logloss"
        p["eval_metric"] = "AUC"
        p["verbose"] = 0
        p["allow_writing_files"] = False
        p["thread_count"] = -1

        # Handle bootstrap_type for subsample compatibility
        if "bootstrap_type" not in p:
            p["bootstrap_type"] = "Bayesian"
        # Bayesian bootstrap doesn't support subsample
        if p.get("bootstrap_type") == "Bayesian":
            p.pop("subsample", None)

        model = CatBoostClassifier(**p, cat_features=cat_indices)
        model.fit(
            X_tr, y.iloc[tr_idx],
            eval_set=(X_va, y.iloc[va_idx]),
            early_stopping_rounds=60,
            use_best_model=True,
        )
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1] / N_FOLDS
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(score)
        print(f"    {name} fold {fold:02d}: {score:.5f}", flush=True)

    auc = roc_auc_score(y, oof)
    print(f"  >> {name} OOF AUC: {auc:.6f}", flush=True)
    return {"name": name, "oof": oof, "test_pred": test_pred, "auc": auc, "fold_scores": fold_scores}


def train_lgbm(
    X: pd.DataFrame, X_test: pd.DataFrame, y: pd.Series,
    params: dict, name: str = "lightgbm",
) -> dict:
    """10-fold LightGBM with fold-level target encoding."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores = []

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

        model = LGBMClassifier(**p, random_state=2000 + fold, n_jobs=-1)
        model.fit(
            X_tr, y.iloc[tr_idx],
            eval_set=[(X_va, y.iloc[va_idx])],
            callbacks=[early_stopping(60, verbose=False), log_evaluation(0)],
        )
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1] / N_FOLDS
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(score)
        print(f"    {name} fold {fold:02d}: {score:.5f}", flush=True)

    auc = roc_auc_score(y, oof)
    print(f"  >> {name} OOF AUC: {auc:.6f}", flush=True)
    return {"name": name, "oof": oof, "test_pred": test_pred, "auc": auc, "fold_scores": fold_scores}


def train_xgb(
    X: pd.DataFrame, X_test: pd.DataFrame, y: pd.Series,
    params: dict, name: str = "xgboost",
) -> dict:
    """10-fold XGBoost with fold-level target encoding."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores = []

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
        es = p.pop("early_stopping_rounds", 60)
        p["early_stopping_rounds"] = es

        model = XGBClassifier(**p, random_state=3000 + fold, n_jobs=-1)
        model.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1] / N_FOLDS
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(score)
        print(f"    {name} fold {fold:02d}: {score:.5f}", flush=True)

    auc = roc_auc_score(y, oof)
    print(f"  >> {name} OOF AUC: {auc:.6f}", flush=True)
    return {"name": name, "oof": oof, "test_pred": test_pred, "auc": auc, "fold_scores": fold_scores}


def train_extratrees(
    X: pd.DataFrame, X_test: pd.DataFrame, y: pd.Series,
    params: dict, name: str = "extratrees",
) -> dict:
    """10-fold ExtraTrees with fold-level target encoding."""
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores = []

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

        model = ExtraTreesClassifier(**p, random_state=4000 + fold)
        model.fit(X_tr, y.iloc[tr_idx])
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1] / N_FOLDS
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(score)
        print(f"    {name} fold {fold:02d}: {score:.5f}", flush=True)

    auc = roc_auc_score(y, oof)
    print(f"  >> {name} OOF AUC: {auc:.6f}", flush=True)
    return {"name": name, "oof": oof, "test_pred": test_pred, "auc": auc, "fold_scores": fold_scores}


# ===========================================================================
# Blending & Stacking
# ===========================================================================
def build_blends_and_stack(
    candidates: list[dict], y: pd.Series, test_ids: pd.Series,
) -> list[dict]:
    """Create blends and a stacked meta-learner from L0 model OOFs."""
    blends = []
    names = [c["name"] for c in candidates]
    oof_prob = np.column_stack([c["oof"] for c in candidates])
    test_prob = np.column_stack([c["test_pred"] for c in candidates])
    oof_rank = np.column_stack([_rank01(c["oof"]) for c in candidates])
    test_rank = np.column_stack([_rank01(c["test_pred"]) for c in candidates])

    def _add(name, oof_pred, test_pred):
        auc = roc_auc_score(y, oof_pred)
        blends.append({"name": name, "oof": oof_pred, "test_pred": test_pred, "auc": auc, "fold_scores": []})
        print(f"  >> {name} OOF AUC: {auc:.6f}", flush=True)

    # 1. Simple probability mean
    _add("blend_prob_mean", oof_prob.mean(axis=1), test_prob.mean(axis=1))

    # 2. Simple rank mean
    _add("blend_rank_mean", oof_rank.mean(axis=1), test_rank.mean(axis=1))

    # 3. Top-3 rank mean
    top3 = np.argsort([c["auc"] for c in candidates])[-3:]
    _add("blend_rank_top3", oof_rank[:, top3].mean(axis=1), test_rank[:, top3].mean(axis=1))

    # 4. Top-2 rank mean
    top2 = np.argsort([c["auc"] for c in candidates])[-2:]
    _add("blend_rank_top2", oof_rank[:, top2].mean(axis=1), test_rank[:, top2].mean(axis=1))

    # 5. Weighted rank blend (conservative: few random trials to avoid overfit)
    rng = np.random.default_rng(42)
    best_auc = -1.0
    best_w = None
    for _ in range(5000):  # Much fewer than 50k to prevent OOF overfit
        w = rng.dirichlet(np.ones(len(candidates)))
        auc = roc_auc_score(y, oof_rank @ w)
        if auc > best_auc:
            best_auc = auc
            best_w = w
    _add("blend_rank_weighted", oof_rank @ best_w, test_rank @ best_w)

    # 6. Level-2 Logistic Regression stack on probabilities
    meta_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2026)
    meta_oof = np.zeros(len(y))
    for tr_idx, va_idx in meta_cv.split(oof_prob, y):
        meta = LogisticRegressionCV(
            cv=5, scoring="roc_auc", max_iter=5000, Cs=20,
            penalty="l2", n_jobs=-1, random_state=42,
        )
        meta.fit(oof_prob[tr_idx], y.iloc[tr_idx])
        meta_oof[va_idx] = meta.predict_proba(oof_prob[va_idx])[:, 1]

    # Final meta for test
    final_meta = LogisticRegressionCV(
        cv=10, scoring="roc_auc", max_iter=5000, Cs=20,
        penalty="l2", n_jobs=-1, random_state=42,
    )
    final_meta.fit(oof_prob, y)
    meta_test = final_meta.predict_proba(test_prob)[:, 1]
    _add("stack_lr_meta", meta_oof, meta_test)

    # 7. Level-2 LR stack on RANK-transformed probabilities
    meta_oof_r = np.zeros(len(y))
    for tr_idx, va_idx in meta_cv.split(oof_rank, y):
        meta = LogisticRegressionCV(
            cv=5, scoring="roc_auc", max_iter=5000, Cs=20,
            penalty="l2", n_jobs=-1, random_state=42,
        )
        meta.fit(oof_rank[tr_idx], y.iloc[tr_idx])
        meta_oof_r[va_idx] = meta.predict_proba(oof_rank[va_idx])[:, 1]

    final_meta_r = LogisticRegressionCV(
        cv=10, scoring="roc_auc", max_iter=5000, Cs=20,
        penalty="l2", n_jobs=-1, random_state=42,
    )
    final_meta_r.fit(oof_rank, y)
    meta_test_r = final_meta_r.predict_proba(test_rank)[:, 1]
    _add("stack_lr_rank_meta", meta_oof_r, meta_test_r)

    return blends


# ===========================================================================
# Save & Report
# ===========================================================================
def save_all_submissions(
    candidates: list[dict],
    blends: list[dict],
    test_ids: pd.Series,
) -> pd.DataFrame:
    """Save all submissions and return summary DataFrame."""
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = candidates + blends
    rows = []

    for r in all_results:
        path = SUB_DIR / f"{r['name']}.csv"
        sub = pd.DataFrame({"Id": test_ids, "Drafted": r["test_pred"]})
        sub.to_csv(path, index=False)

        rows.append({
            "name": r["name"],
            "oof_auc": r["auc"],
            "fold_mean": float(np.mean(r["fold_scores"])) if r["fold_scores"] else np.nan,
            "fold_std": float(np.std(r["fold_scores"])) if r["fold_scores"] else np.nan,
            "file": str(path.name),
        })

    summary = pd.DataFrame(rows).sort_values("oof_auc", ascending=False)
    summary.to_csv(RESULTS_DIR / "final_summary.csv", index=False)

    # Save the best as the primary submission
    best = all_results[int(np.argmax([r["auc"] for r in all_results]))]
    primary_path = SUB_DIR / "BEST_SUBMISSION.csv"
    pd.DataFrame({"Id": test_ids, "Drafted": best["test_pred"]}).to_csv(primary_path, index=False)

    print(f"\n{'='*60}")
    print(f"BEST candidate: {best['name']} (OOF AUC: {best['auc']:.6f})")
    print(f"Saved to: {primary_path}")
    print(f"{'='*60}\n")

    return summary

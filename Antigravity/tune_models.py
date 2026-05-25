"""
Antigravity — Optuna Hyperparameter Tuning for NFL Draft Models.

Tunes CatBoost, LightGBM, XGBoost, and ExtraTrees on the advanced feature matrix.
Uses 5-fold inner CV for speed during tuning, 10-fold for final evaluation.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import category_encoders as ce

from features import CAT_COLS, LOW_CARD_CATS, add_target_encoding_fold

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PARAMS_DIR = Path(__file__).parent / "tuned_params"


def _numeric_fold(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    with_target_encoding: bool = True,
) -> tuple:
    """Prepare numeric-only fold matrices with LOO target encoding."""
    if with_target_encoding:
        X_tr, X_va, X_te = add_target_encoding_fold(X, y, X_test, tr_idx, va_idx)
    else:
        X_tr = X.iloc[tr_idx].copy()
        X_va = X.iloc[va_idx].copy()
        X_te = X_test.copy()

    # Drop raw categorical columns for non-CatBoost models
    drop_cols = [c for c in CAT_COLS if c in X_tr.columns]
    X_tr = X_tr.drop(columns=drop_cols)
    X_va = X_va.drop(columns=drop_cols)
    X_te = X_te.drop(columns=drop_cols)

    # Also add LOO encoding for low-card cats via category_encoders
    # (provides a second encoding signal alongside our Bayesian TE)
    loo_cols = [c for c in LOW_CARD_CATS if c not in X_tr.columns]
    # Already dropped, so skip this

    X_tr = X_tr.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_va = X_va.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_te = X_te.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X_tr, X_va, X_te


def _catboost_fold(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
) -> tuple:
    """Prepare CatBoost fold matrices (keeps categoricals native)."""
    X_tr = X.iloc[tr_idx].copy()
    X_va = X.iloc[va_idx].copy()
    X_te = X_test.copy()

    for col in CAT_COLS:
        if col in X_tr.columns:
            X_tr[col] = X_tr[col].astype("category")
            X_va[col] = X_va[col].astype("category")
            X_te[col] = X_te[col].astype("category")

    return X_tr, X_va, X_te


# ===== CatBoost Tuning =====
def _catboost_objective(trial, X, X_test, y, n_splits=5):
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])

    params = {
        "iterations": trial.suggest_int("iterations", 500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "depth": trial.suggest_int("depth", 3, 8),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": 0,
        "allow_writing_files": False,
        "thread_count": -1,
        "bootstrap_type": bootstrap_type,
    }

    # Bayesian bootstrap uses bagging_temperature, not subsample
    # Bernoulli/MVS use subsample, not bagging_temperature
    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        X_tr, X_va, X_te = _catboost_fold(X, X_test, tr_idx, va_idx)
        cat_indices = [i for i, c in enumerate(X_tr.columns) if c in CAT_COLS]

        model = CatBoostClassifier(**params, random_seed=42 + fold, cat_features=cat_indices)
        model.fit(
            X_tr, y.iloc[tr_idx],
            eval_set=(X_va, y.iloc[va_idx]),
            early_stopping_rounds=50,
            use_best_model=True,
        )
        oof[va_idx] = model.predict_proba(X_va)[:, 1]

    return roc_auc_score(y, oof)


# ===== LightGBM Tuning =====
def _lgbm_objective(trial, X, X_test, y, n_splits=5):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 5.0),
        "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.3, 1.0),
        "objective": "binary",
        "metric": "auc",
        "verbose": -1,
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        X_tr, X_va, X_te = _numeric_fold(X, X_test, y, tr_idx, va_idx)

        model = LGBMClassifier(**params, random_state=2000 + fold, n_jobs=-1)
        model.fit(
            X_tr, y.iloc[tr_idx],
            eval_set=[(X_va, y.iloc[va_idx])],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
        )
        oof[va_idx] = model.predict_proba(X_va)[:, 1]

    return roc_auc_score(y, oof)


# ===== XGBoost Tuning =====
def _xgb_objective(trial, X, X_test, y, n_splits=5):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "early_stopping_rounds": 50,
        "verbosity": 0,
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        X_tr, X_va, X_te = _numeric_fold(X, X_test, y, tr_idx, va_idx)

        model = XGBClassifier(**params, random_state=3000 + fold, n_jobs=-1)
        model.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]

    return roc_auc_score(y, oof)


# ===== ExtraTrees Tuning =====
def _et_objective(trial, X, X_test, y, n_splits=5):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "max_depth": trial.suggest_int("max_depth", 5, 25),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "n_jobs": -1,
    }
    cw = trial.suggest_categorical("class_weight", ["balanced", "none"])
    if cw == "none":
        params["class_weight"] = None
    else:
        params["class_weight"] = cw

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        X_tr, X_va, X_te = _numeric_fold(X, X_test, y, tr_idx, va_idx)

        model = ExtraTreesClassifier(**params, random_state=4000 + fold)
        model.fit(X_tr, y.iloc[tr_idx])
        oof[va_idx] = model.predict_proba(X_va)[:, 1]

    return roc_auc_score(y, oof)


# ===== Main tuning orchestrator =====
def tune_all(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 80,
    force_retune: bool = False,
) -> dict:
    """
    Tune all model families with Optuna.
    
    Returns dict of {model_name: best_params}.
    Caches results to tuned_params/ directory.
    """
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    all_params = {}

    models = [
        ("catboost", _catboost_objective),
        ("lightgbm", _lgbm_objective),
        ("xgboost", _xgb_objective),
        ("extratrees", _et_objective),
    ]

    for name, objective_fn in models:
        cache_path = PARAMS_DIR / f"{name}_best.json"

        if cache_path.exists() and not force_retune:
            print(f"  [{name}] Loading cached params from {cache_path}")
            with open(cache_path) as f:
                result = json.load(f)
            all_params[name] = result
            continue

        print(f"  [{name}] Starting Optuna tuning ({n_trials} trials)...", flush=True)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            lambda trial: objective_fn(trial, X_train, X_test, y),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        result = {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": n_trials,
        }
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"  [{name}] Best OOF AUC: {study.best_value:.6f}")
        all_params[name] = result

    return all_params


if __name__ == "__main__":
    from features import build_features

    train = pd.read_csv("Data/train.csv")
    test = pd.read_csv("Data/test.csv")
    y = train["Drafted"].reset_index(drop=True)

    X_train, X_test = build_features(train, test)
    print(f"Feature matrix: {X_train.shape[1]} features")

    results = tune_all(X_train, X_test, y, n_trials=80)

    for name, result in results.items():
        print(f"\n{name}: OOF AUC = {result['best_value']:.6f}")
        print(f"  Params: {result['best_params']}")

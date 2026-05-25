from __future__ import annotations

import json
from pathlib import Path

import category_encoders as ce
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
OUT_DIR = ROOT / "legal_tournament"
SUB_DIR = OUT_DIR / "submissions"
RESULT_DIR = OUT_DIR / "results"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
ACTIVE_SUBMISSION_PATH = DATA_DIR / "submission.csv"

CAT_COLS = ["School", "Player_Type", "Position_Type", "Position"]
LOW_CARD_CATS = ["Player_Type", "Position_Type", "Position"]
PHYSICAL = [
    "Height",
    "Weight",
    "Sprint_40yd",
    "Vertical_Jump",
    "Bench_Press_Reps",
    "Broad_Jump",
    "Agility_3cone",
    "Shuttle",
    "Age",
]
COMBINE = [
    "Sprint_40yd",
    "Vertical_Jump",
    "Bench_Press_Reps",
    "Broad_Jump",
    "Agility_3cone",
    "Shuttle",
]


def ensure_dirs() -> None:
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def save_submission(name: str, test_ids: pd.Series, preds: np.ndarray) -> Path:
    path = SUB_DIR / f"{name}.csv"
    submission = pd.DataFrame({"Id": test_ids, "Drafted": preds})
    submission.to_csv(path, index=False)
    return path


def print_score(name: str, y: pd.Series, oof: np.ndarray) -> float:
    auc = roc_auc_score(y, oof)
    print(f"{name:32s} OOF ROC-AUC: {auc:.6f}", flush=True)
    return auc


def rank01(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy() / len(values)


def build_feature_frame(train: pd.DataFrame, test: pd.DataFrame, rich: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.concat([train.drop(columns=["Drafted"]), test], axis=0).reset_index(drop=True)

    for col in CAT_COLS:
        df[col] = df[col].fillna("Unknown").astype(str)

    for col in PHYSICAL:
        if df[col].isnull().sum() > 0:
            df[f"{col}_Is_Missing"] = df[col].isnull().astype(int)

    df["Combine_Missing_Count"] = df[COMBINE].isnull().sum(axis=1)
    df["Combine_Completed_Count"] = df[COMBINE].notnull().sum(axis=1)
    df["Did_Any_Agility"] = df[["Agility_3cone", "Shuttle"]].notnull().any(axis=1).astype(int)
    df["Did_All_Explosive"] = df[["Vertical_Jump", "Broad_Jump"]].notnull().all(axis=1).astype(int)

    for col in PHYSICAL:
        df[col] = df[col].fillna(df.groupby("Position")[col].transform("median"))
        df[col] = df[col].fillna(df.groupby("Position_Type")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

    weight_lbs = df["Weight"] * 2.20462
    height_in = df["Height"] * 39.3701
    safe_40 = df["Sprint_40yd"].replace(0, np.nan)
    safe_weight = weight_lbs.replace(0, np.nan)

    df["Weight_lbs"] = weight_lbs
    df["Height_inches"] = height_in
    df["BMI_metric"] = df["Weight"] / (df["Height"] ** 2)
    df["BMI_football"] = (weight_lbs / (height_in**2)) * 703
    df["Speed_Weight_Ratio"] = weight_lbs / safe_40
    df["Speed_Score"] = (weight_lbs * 200) / (safe_40**4)
    df["Explosiveness_Index"] = df["Vertical_Jump"] * df["Broad_Jump"]
    df["Burst_Score"] = df["Vertical_Jump"] + df["Broad_Jump"]
    df["Power_Score"] = df["Bench_Press_Reps"] * weight_lbs
    df["Power_Burst"] = df["Bench_Press_Reps"] * df["Broad_Jump"]
    df["Agility_Total"] = df["Agility_3cone"] + df["Shuttle"]
    df["Agility_Speed_Ratio"] = df["Agility_Total"] / safe_40
    df["Broad_per_lb"] = df["Broad_Jump"] / safe_weight
    df["Vertical_per_lb"] = df["Vertical_Jump"] / safe_weight

    for col in CAT_COLS:
        df[f"{col}_freq_all"] = df[col].map(df[col].value_counts())

    combo_groups = [
        ["School", "Position"],
        ["School", "Position_Type"],
        ["Year", "Position"],
        ["Year", "Position_Type"],
        ["Position", "Player_Type"],
    ]
    if rich:
        combo_groups.extend([["Year", "School"], ["School", "Player_Type"]])

    for cols in combo_groups:
        name = "_x_".join(cols)
        df[f"{name}_freq_all"] = df.groupby(cols)["Id"].transform("count")

    rel_metrics = [
        "Age",
        "Height",
        "Weight",
        "Sprint_40yd",
        "Vertical_Jump",
        "Bench_Press_Reps",
        "Broad_Jump",
        "Agility_3cone",
        "Shuttle",
        "BMI_metric",
        "BMI_football",
        "Speed_Weight_Ratio",
        "Speed_Score",
        "Explosiveness_Index",
        "Burst_Score",
        "Power_Score",
        "Agility_Total",
    ]
    rel_groups = ["Position"]
    if rich:
        rel_groups.extend(["Position_Type", "Player_Type"])

    for group in rel_groups:
        for col in rel_metrics:
            mean = df.groupby(group)[col].transform("mean")
            std = df.groupby(group)[col].transform("std").replace(0, np.nan)
            df[f"{col}_vs_{group}_mean"] = df[col] - mean
            if rich:
                df[f"{col}_z_{group}"] = (df[col] - mean) / std

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_train = df.iloc[: len(train)].drop(columns=["Id"]).reset_index(drop=True)
    X_test = df.iloc[len(train) :].drop(columns=["Id"]).reset_index(drop=True)
    return X_train, X_test


def build_clean_catboost_matrix(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact compact matrix used by the clean audit notebook."""
    df = pd.concat([train.drop(columns=["Drafted"]), test], axis=0).reset_index(drop=True)

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
        df[col] = df[col].fillna("Unknown").astype(str)

    X_train = df.iloc[: len(train)].drop(columns=["Id"]).reset_index(drop=True)
    X_test = df.iloc[len(train) :].drop(columns=["Id"]).reset_index(drop=True)
    return X_train, X_test


def as_catboost_frame(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in CAT_COLS:
        out[col] = out[col].astype("category")
    return out


def numeric_fold_matrices(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    scale: bool = False,
) -> tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]:
    X_tr = X.iloc[tr_idx].copy()
    X_va = X.iloc[va_idx].copy()
    X_te = X_test.copy()
    y_tr = y.iloc[tr_idx]

    encoder = ce.LeaveOneOutEncoder(cols=CAT_COLS, sigma=0.05)
    X_tr_te = encoder.fit_transform(X_tr[CAT_COLS], y_tr).add_suffix("_loo")
    X_va_te = encoder.transform(X_va[CAT_COLS]).add_suffix("_loo")
    X_te_te = encoder.transform(X_te[CAT_COLS]).add_suffix("_loo")

    X_tr_oh = pd.get_dummies(X_tr[LOW_CARD_CATS], prefix=LOW_CARD_CATS)
    X_va_oh = pd.get_dummies(X_va[LOW_CARD_CATS], prefix=LOW_CARD_CATS)
    X_te_oh = pd.get_dummies(X_te[LOW_CARD_CATS], prefix=LOW_CARD_CATS)
    X_va_oh = X_va_oh.reindex(columns=X_tr_oh.columns, fill_value=0)
    X_te_oh = X_te_oh.reindex(columns=X_tr_oh.columns, fill_value=0)

    num_cols = [c for c in X.columns if c not in CAT_COLS]
    X_tr_num = X_tr[num_cols].reset_index(drop=True)
    X_va_num = X_va[num_cols].reset_index(drop=True)
    X_te_num = X_te[num_cols].reset_index(drop=True)

    X_tr_final = pd.concat([X_tr_num, X_tr_te.reset_index(drop=True), X_tr_oh.reset_index(drop=True)], axis=1)
    X_va_final = pd.concat([X_va_num, X_va_te.reset_index(drop=True), X_va_oh.reset_index(drop=True)], axis=1)
    X_te_final = pd.concat([X_te_num, X_te_te.reset_index(drop=True), X_te_oh.reset_index(drop=True)], axis=1)

    X_tr_final = X_tr_final.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_va_final = X_va_final.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_te_final = X_te_final.replace([np.inf, -np.inf], np.nan).fillna(0)

    if not scale:
        return X_tr_final, X_va_final, X_te_final

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_final)
    X_va_scaled = scaler.transform(X_va_final)
    X_te_scaled = scaler.transform(X_te_final)
    return X_tr_scaled, X_va_scaled, X_te_scaled


def run_catboost_cv(
    name: str,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    test_ids: pd.Series,
    params: dict,
    n_splits: int = 10,
) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores = []

    X_cb = as_catboost_frame(X)
    X_test_cb = as_catboost_frame(X_test)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_cb, y), start=1):
        fold_params = params.copy()
        fold_params["random_seed"] = 1000 + fold
        model = CatBoostClassifier(**fold_params, cat_features=CAT_COLS)
        model.fit(
            X_cb.iloc[tr_idx],
            y.iloc[tr_idx],
            eval_set=(X_cb.iloc[va_idx], y.iloc[va_idx]),
            early_stopping_rounds=60,
            use_best_model=True,
        )
        oof[va_idx] = model.predict_proba(X_cb.iloc[va_idx])[:, 1]
        test_pred += model.predict_proba(X_test_cb)[:, 1] / n_splits
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(score)
        print(f"  {name} fold {fold:02d}: {score:.5f}", flush=True)

    path = save_submission(name, test_ids, test_pred)
    return {
        "name": name,
        "oof": oof,
        "test_pred": test_pred,
        "auc": print_score(name, y, oof),
        "fold_scores": fold_scores,
        "submission": str(path.relative_to(ROOT)),
    }


def run_lgbm_cv(
    name: str,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    test_ids: pd.Series,
    params: dict,
    n_splits: int = 10,
) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_va, X_te = numeric_fold_matrices(X, X_test, y, tr_idx, va_idx)
        model = LGBMClassifier(**params, random_state=2000 + fold, n_jobs=-1, verbose=-1)
        model.fit(
            X_tr,
            y.iloc[tr_idx],
            eval_set=[(X_va, y.iloc[va_idx])],
            callbacks=[early_stopping(80, verbose=False), log_evaluation(0)],
        )
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1] / n_splits
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(score)
        print(f"  {name} fold {fold:02d}: {score:.5f}", flush=True)

    path = save_submission(name, test_ids, test_pred)
    return {
        "name": name,
        "oof": oof,
        "test_pred": test_pred,
        "auc": print_score(name, y, oof),
        "fold_scores": fold_scores,
        "submission": str(path.relative_to(ROOT)),
    }


def run_xgb_cv(
    name: str,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    test_ids: pd.Series,
    params: dict,
    n_splits: int = 10,
) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_va, X_te = numeric_fold_matrices(X, X_test, y, tr_idx, va_idx)
        model = XGBClassifier(**params, random_state=3000 + fold, n_jobs=-1)
        model.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1] / n_splits
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(score)
        print(f"  {name} fold {fold:02d}: {score:.5f}", flush=True)

    path = save_submission(name, test_ids, test_pred)
    return {
        "name": name,
        "oof": oof,
        "test_pred": test_pred,
        "auc": print_score(name, y, oof),
        "fold_scores": fold_scores,
        "submission": str(path.relative_to(ROOT)),
    }


def run_sklearn_cv(
    name: str,
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    test_ids: pd.Series,
    model_factory,
    n_splits: int = 10,
    scale: bool = False,
) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        X_tr, X_va, X_te = numeric_fold_matrices(X, X_test, y, tr_idx, va_idx, scale=scale)
        model = model_factory(4000 + fold)
        model.fit(X_tr, y.iloc[tr_idx])
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_pred += model.predict_proba(X_te)[:, 1] / n_splits
        score = roc_auc_score(y.iloc[va_idx], oof[va_idx])
        fold_scores.append(score)
        print(f"  {name} fold {fold:02d}: {score:.5f}", flush=True)

    path = save_submission(name, test_ids, test_pred)
    return {
        "name": name,
        "oof": oof,
        "test_pred": test_pred,
        "auc": print_score(name, y, oof),
        "fold_scores": fold_scores,
        "submission": str(path.relative_to(ROOT)),
    }


def make_blends(candidates: list[dict], y: pd.Series, test_ids: pd.Series) -> list[dict]:
    blends = []
    names = [c["name"] for c in candidates]
    oof_rank = np.column_stack([rank01(c["oof"]) for c in candidates])
    test_rank = np.column_stack([rank01(c["test_pred"]) for c in candidates])
    oof_prob = np.column_stack([c["oof"] for c in candidates])
    test_prob = np.column_stack([c["test_pred"] for c in candidates])

    def add_blend(name: str, oof_pred: np.ndarray, test_pred: np.ndarray, weights: np.ndarray | None = None) -> None:
        path = save_submission(name, test_ids, test_pred)
        auc = print_score(name, y, oof_pred)
        blends.append(
            {
                "name": name,
                "oof": oof_pred,
                "test_pred": test_pred,
                "auc": auc,
                "fold_scores": [],
                "submission": str(path.relative_to(ROOT)),
                "weights": dict(zip(names, weights.round(6).tolist())) if weights is not None else {},
            }
        )

    add_blend("blend_prob_mean_all", oof_prob.mean(axis=1), test_prob.mean(axis=1))
    add_blend("blend_rank_mean_all", oof_rank.mean(axis=1), test_rank.mean(axis=1))

    top_idx = np.argsort([c["auc"] for c in candidates])[-4:]
    add_blend(
        "blend_rank_mean_top4",
        oof_rank[:, top_idx].mean(axis=1),
        test_rank[:, top_idx].mean(axis=1),
        np.eye(len(candidates))[top_idx].sum(axis=0) / len(top_idx),
    )

    rng = np.random.default_rng(42)
    best_auc = -np.inf
    best_w = None
    for _ in range(50000):
        w = rng.dirichlet(np.ones(len(candidates)))
        auc = roc_auc_score(y, oof_rank @ w)
        if auc > best_auc:
            best_auc = auc
            best_w = w
    add_blend("blend_rank_random_search", oof_rank @ best_w, test_rank @ best_w, best_w)

    meta_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2026)
    meta_oof = np.zeros(len(y))
    for tr_idx, va_idx in meta_cv.split(oof_prob, y):
        meta = LogisticRegressionCV(cv=5, scoring="roc_auc", max_iter=3000, n_jobs=-1, random_state=42)
        meta.fit(oof_prob[tr_idx], y.iloc[tr_idx])
        meta_oof[va_idx] = meta.predict_proba(oof_prob[va_idx])[:, 1]
    final_meta = LogisticRegressionCV(cv=10, scoring="roc_auc", max_iter=3000, n_jobs=-1, random_state=42)
    final_meta.fit(oof_prob, y)
    meta_test = final_meta.predict_proba(test_prob)[:, 1]
    add_blend("blend_lr_meta_oof", meta_oof, meta_test)

    return blends


def main() -> None:
    ensure_dirs()
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y = train["Drafted"].reset_index(drop=True)
    test_ids = test["Id"].reset_index(drop=True)

    print("=== Legal NFL Modeling Tournament ===", flush=True)
    print(f"Train: {train.shape} | Test: {test.shape}", flush=True)
    print(f"Target: {int((y == 1).sum())} drafted / {int((y == 0).sum())} undrafted", flush=True)

    X_clean, X_test_clean = build_clean_catboost_matrix(train, test)
    X_legacy, X_test_legacy = build_feature_frame(train, test, rich=False)
    X_rich, X_test_rich = build_feature_frame(train, test, rich=True)
    print(
        f"Clean features: {X_clean.shape[1]} | Legacy features: {X_legacy.shape[1]} | Rich features: {X_rich.shape[1]}",
        flush=True,
    )

    candidates: list[dict] = []

    cat_params = {
        "iterations": 1183,
        "learning_rate": 0.010492928203031996,
        "depth": 4,
        "l2_leaf_reg": 4.5594175959311425,
        "min_data_in_leaf": 7,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": 0,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    candidates.append(run_catboost_cv("cat_clean_10fold", X_clean, X_test_clean, y, test_ids, cat_params))
    candidates.append(run_catboost_cv("cat_legacy_10fold", X_legacy, X_test_legacy, y, test_ids, cat_params))

    cat_rich_params = {
        "iterations": 900,
        "learning_rate": 0.018,
        "depth": 5,
        "l2_leaf_reg": 6.0,
        "min_data_in_leaf": 18,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "verbose": 0,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    candidates.append(run_catboost_cv("cat_rich_10fold", X_rich, X_test_rich, y, test_ids, cat_rich_params))

    lgbm_params = {
        "n_estimators": 1800,
        "learning_rate": 0.012,
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 35,
        "subsample": 0.82,
        "colsample_bytree": 0.70,
        "reg_alpha": 0.2,
        "reg_lambda": 4.0,
        "objective": "binary",
        "metric": "auc",
    }
    candidates.append(run_lgbm_cv("lgbm_rich_10fold", X_rich, X_test_rich, y, test_ids, lgbm_params))

    xgb_params = {
        "n_estimators": 1400,
        "learning_rate": 0.012,
        "max_depth": 3,
        "min_child_weight": 8,
        "subsample": 0.82,
        "colsample_bytree": 0.72,
        "reg_alpha": 0.1,
        "reg_lambda": 5.0,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "early_stopping_rounds": 80,
        "tree_method": "hist",
    }
    candidates.append(run_xgb_cv("xgb_rich_10fold", X_rich, X_test_rich, y, test_ids, xgb_params))

    candidates.append(
        run_sklearn_cv(
            "extra_trees_rich_10fold",
            X_rich,
            X_test_rich,
            y,
            test_ids,
            lambda seed: ExtraTreesClassifier(
                n_estimators=900,
                max_features="sqrt",
                min_samples_leaf=4,
                class_weight=None,
                random_state=seed,
                n_jobs=-1,
            ),
        )
    )

    candidates.append(
        run_sklearn_cv(
            "rf_rich_10fold",
            X_rich,
            X_test_rich,
            y,
            test_ids,
            lambda seed: RandomForestClassifier(
                n_estimators=900,
                max_depth=9,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=seed,
                n_jobs=-1,
            ),
        )
    )

    blends = make_blends(candidates, y, test_ids)
    all_results = candidates + blends

    summary_rows = []
    for result in all_results:
        row = {
            "name": result["name"],
            "auc": result["auc"],
            "submission": result["submission"],
            "fold_mean": float(np.mean(result["fold_scores"])) if result["fold_scores"] else np.nan,
            "fold_std": float(np.std(result["fold_scores"])) if result["fold_scores"] else np.nan,
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("auc", ascending=False)
    summary_path = RESULT_DIR / "tournament_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n=== Tournament Summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)

    best = all_results[int(np.argmax([r["auc"] for r in all_results]))]
    pd.DataFrame({"Id": test_ids, "Drafted": best["test_pred"]}).to_csv(ACTIVE_SUBMISSION_PATH, index=False)
    print(f"\nBest local-OOF candidate: {best['name']} ({best['auc']:.6f})", flush=True)
    print(f"Copied to active submission: {ACTIVE_SUBMISSION_PATH.relative_to(ROOT)}", flush=True)

    payload = {
        "best": {"name": best["name"], "auc": best["auc"], "submission": best["submission"]},
        "results": summary_rows,
    }
    (RESULT_DIR / "tournament_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

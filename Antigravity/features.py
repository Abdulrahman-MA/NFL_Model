"""
Antigravity — Advanced Feature Engineering for NFL Draft Prediction.

All features are computed from train.csv and test.csv only.
Target-aware features use leak-free fold-based computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PHYSICAL = [
    "Height", "Weight", "Sprint_40yd", "Vertical_Jump",
    "Bench_Press_Reps", "Broad_Jump", "Agility_3cone", "Shuttle", "Age",
]
COMBINE_DRILLS = [
    "Sprint_40yd", "Vertical_Jump", "Bench_Press_Reps",
    "Broad_Jump", "Agility_3cone", "Shuttle",
]
CAT_COLS = ["School", "Player_Type", "Position_Type", "Position"]
LOW_CARD_CATS = ["Player_Type", "Position_Type", "Position"]


def _safe_div(a, b, fill=0.0):
    """Division with zero/nan protection."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b
    return np.where(np.isfinite(result), result, fill)


# ---------------------------------------------------------------------------
# 1. Missingness features
# ---------------------------------------------------------------------------
def _add_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    """Detailed missingness pattern features."""
    # Per-column missing flags
    for col in PHYSICAL:
        if df[col].isnull().any():
            df[f"{col}_missing"] = df[col].isnull().astype(np.int8)

    # Aggregate missingness
    df["n_combine_missing"] = df[COMBINE_DRILLS].isnull().sum(axis=1)
    df["n_combine_present"] = df[COMBINE_DRILLS].notnull().sum(axis=1)
    df["n_physical_missing"] = df[PHYSICAL].isnull().sum(axis=1)
    df["pct_combine_present"] = df["n_combine_present"] / len(COMBINE_DRILLS)

    # Specific drill group patterns
    df["did_speed"] = df["Sprint_40yd"].notnull().astype(np.int8)
    df["did_explosiveness"] = (
        df[["Vertical_Jump", "Broad_Jump"]].notnull().all(axis=1).astype(np.int8)
    )
    df["did_agility"] = (
        df[["Agility_3cone", "Shuttle"]].notnull().any(axis=1).astype(np.int8)
    )
    df["did_all_agility"] = (
        df[["Agility_3cone", "Shuttle"]].notnull().all(axis=1).astype(np.int8)
    )
    df["did_bench"] = df["Bench_Press_Reps"].notnull().astype(np.int8)
    df["did_all_drills"] = (
        df[COMBINE_DRILLS].notnull().all(axis=1).astype(np.int8)
    )
    df["has_age"] = df["Age"].notnull().astype(np.int8)

    # Missingness × position interaction (which drills a position typically skips)
    for pos_col in ["Position", "Position_Type"]:
        group_miss_rate = df.groupby(pos_col)["n_combine_missing"].transform("mean")
        df[f"miss_vs_{pos_col}_avg"] = df["n_combine_missing"] - group_miss_rate

    return df


# ---------------------------------------------------------------------------
# 2. Imputation
# ---------------------------------------------------------------------------
def _impute(df: pd.DataFrame) -> pd.DataFrame:
    """Position-aware median imputation with fallback chain."""
    for col in PHYSICAL:
        df[col] = df[col].fillna(df.groupby("Position")[col].transform("median"))
        df[col] = df[col].fillna(df.groupby("Position_Type")[col].transform("median"))
        df[col] = df[col].fillna(df.groupby("Player_Type")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())
    return df


# ---------------------------------------------------------------------------
# 3. Core athletic features
# ---------------------------------------------------------------------------
def _add_core_athletic(df: pd.DataFrame) -> pd.DataFrame:
    """Combine-derived athletic metrics."""
    h = df["Height"]
    w = df["Weight"]
    s40 = df["Sprint_40yd"].replace(0, np.nan)
    vj = df["Vertical_Jump"]
    bj = df["Broad_Jump"]
    bp = df["Bench_Press_Reps"]
    cone = df["Agility_3cone"]
    shuttle = df["Shuttle"]

    # Body composition
    df["BMI"] = _safe_div(w, h ** 2)
    df["height_weight_ratio"] = _safe_div(h, w)

    # Speed metrics
    df["speed_score"] = _safe_div(w * 200, s40 ** 4)
    df["speed_weight_ratio"] = _safe_div(w, s40)

    # Explosiveness
    df["explosiveness"] = vj * bj
    df["burst_score"] = vj + bj
    df["vertical_broad_ratio"] = _safe_div(vj, bj)
    df["broad_per_kg"] = _safe_div(bj, w)
    df["vertical_per_kg"] = _safe_div(vj, w)

    # Strength
    df["power_score"] = bp * w
    df["bench_per_kg"] = _safe_div(bp, w)
    df["power_burst"] = bp * bj

    # Agility
    df["agility_total"] = cone + shuttle
    df["agility_speed_ratio"] = _safe_div(df["agility_total"], s40)
    df["cone_shuttle_diff"] = cone - shuttle
    df["shuttle_per_kg"] = _safe_div(shuttle, w)

    # Composite scores
    df["athletic_composite"] = (
        _safe_div(vj, vj.median()) +
        _safe_div(bj, bj.median()) +
        _safe_div(bp, bp.median()) +
        _safe_div(s40.median(), s40)  # Inverted — lower is better
    )

    # Age interactions
    df["age_speed"] = df["Age"] * s40
    df["age_squared"] = df["Age"] ** 2

    return df


# ---------------------------------------------------------------------------
# 4. Percentile rank features (non-parametric, within-position)
# ---------------------------------------------------------------------------
def _add_percentile_features(df: pd.DataFrame) -> pd.DataFrame:
    """Within-group percentile ranks for all metrics."""
    metrics = PHYSICAL + [
        "BMI", "speed_score", "speed_weight_ratio", "explosiveness",
        "burst_score", "power_score", "agility_total", "athletic_composite",
    ]

    for group_col in ["Position", "Position_Type"]:
        for col in metrics:
            if col in df.columns:
                df[f"{col}_pctile_{group_col}"] = df.groupby(group_col)[col].rank(
                    pct=True, method="average"
                )

    # Year-level percentiles (how does this player compare to their draft class?)
    for col in metrics:
        if col in df.columns:
            df[f"{col}_pctile_Year"] = df.groupby("Year")[col].rank(
                pct=True, method="average"
            )

    return df


# ---------------------------------------------------------------------------
# 5. Position-relative z-scores and differences
# ---------------------------------------------------------------------------
def _add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deviation from group mean for key metrics."""
    metrics = PHYSICAL + [
        "BMI", "speed_score", "explosiveness", "power_score",
        "burst_score", "agility_total",
    ]

    for group_col in ["Position", "Position_Type", "Year"]:
        for col in metrics:
            if col in df.columns:
                group_mean = df.groupby(group_col)[col].transform("mean")
                group_std = df.groupby(group_col)[col].transform("std").replace(0, 1)
                df[f"{col}_diff_{group_col}"] = df[col] - group_mean
                df[f"{col}_zscore_{group_col}"] = (df[col] - group_mean) / group_std

    return df


# ---------------------------------------------------------------------------
# 6. Frequency / count features
# ---------------------------------------------------------------------------
def _add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Group frequency / count features."""
    for col in CAT_COLS:
        df[f"{col}_freq"] = df.groupby(col)["Id"].transform("count")

    # Combo frequencies
    combos = [
        ["School", "Position"],
        ["School", "Position_Type"],
        ["Year", "Position"],
        ["Year", "Position_Type"],
        ["Position", "Player_Type"],
        ["Year", "School"],
        ["School", "Player_Type"],
    ]
    for cols in combos:
        name = "_x_".join(cols)
        df[f"{name}_freq"] = df.groupby(cols)["Id"].transform("count")

    # School size relative to year (big school = sends more players)
    df["school_per_year"] = df.groupby(["School", "Year"])["Id"].transform("count")
    df["school_total"] = df.groupby("School")["Id"].transform("count")

    return df


# ---------------------------------------------------------------------------
# 7. Target encoding (leak-free, fold-aware)
# ---------------------------------------------------------------------------
def _smoothed_target_encode(
    train_col: pd.Series,
    train_y: pd.Series,
    test_col: pd.Series,
    smoothing: float = 10.0,
    noise: float = 0.01,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bayesian-smoothed leave-one-out target encoding.
    
    For train: leave-one-out (removes self from mean calculation).
    For test: full train statistics.
    Smoothing factor blends toward global mean for rare categories.
    """
    global_mean = train_y.mean()
    rng = np.random.RandomState(seed)

    # Full stats for test encoding
    stats = pd.DataFrame({"col": train_col, "y": train_y})
    group_stats = stats.groupby("col")["y"].agg(["sum", "count"])
    group_stats["smooth_mean"] = (
        (group_stats["sum"] + smoothing * global_mean)
        / (group_stats["count"] + smoothing)
    )

    # Test encoding: map using smoothed means
    test_encoded = test_col.map(group_stats["smooth_mean"]).fillna(global_mean).values

    # Train encoding: leave-one-out with smoothing
    train_encoded = np.full(len(train_col), global_mean)
    for cat, grp in group_stats.iterrows():
        mask = train_col == cat
        n = grp["count"]
        s = grp["sum"]
        if n > 1:
            loo_sum = s - train_y[mask].values
            loo_count = n - 1
            loo_mean = (loo_sum + smoothing * global_mean) / (loo_count + smoothing)
            train_encoded[mask.values] = loo_mean
        else:
            train_encoded[mask.values] = global_mean

    # Add small noise to train to reduce overfitting
    train_encoded += rng.normal(0, noise, len(train_encoded))

    return train_encoded, test_encoded


def add_target_encoding_fold(
    X_train: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    smoothing: float = 10.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add target-encoded features, computed fold-aware.
    
    For training fold: use LOO on training fold only.
    For validation fold: encode using training fold statistics.
    For test: encode using training fold statistics.
    """
    X_tr = X_train.iloc[tr_idx].copy()
    X_va = X_train.iloc[va_idx].copy()
    X_te = X_test.copy()
    y_tr = y.iloc[tr_idx]

    global_mean = y_tr.mean()

    encode_cols = ["School", "Position", "School_x_Position_combo"]

    # Create combo columns
    for df in [X_tr, X_va, X_te]:
        df["School_x_Position_combo"] = df["School"].astype(str) + "_" + df["Position"].astype(str)

    for col in encode_cols:
        # Stats from training fold only
        stats = pd.DataFrame({"col": X_tr[col], "y": y_tr.values})
        group_stats = stats.groupby("col")["y"].agg(["sum", "count"])
        group_stats["smooth_mean"] = (
            (group_stats["sum"] + smoothing * global_mean)
            / (group_stats["count"] + smoothing)
        )

        # Train fold: LOO
        train_encoded = np.full(len(X_tr), global_mean)
        for cat, grp in group_stats.iterrows():
            mask = X_tr[col] == cat
            n = grp["count"]
            s = grp["sum"]
            if n > 1:
                loo_sum = s - y_tr.values[mask.values]
                loo_count = n - 1
                loo_mean = (loo_sum + smoothing * global_mean) / (loo_count + smoothing)
                train_encoded[mask.values] = loo_mean
            else:
                train_encoded[mask.values] = global_mean
        X_tr[f"{col}_te"] = train_encoded

        # Validation fold: direct map
        X_va[f"{col}_te"] = X_va[col].map(group_stats["smooth_mean"]).fillna(global_mean).values

        # Test: direct map
        X_te[f"{col}_te"] = X_te[col].map(group_stats["smooth_mean"]).fillna(global_mean).values

    # Drop combo col
    for df in [X_tr, X_va, X_te]:
        df.drop(columns=["School_x_Position_combo"], inplace=True, errors="ignore")

    return X_tr, X_va, X_te


# ---------------------------------------------------------------------------
# 8. Pairwise interaction features
# ---------------------------------------------------------------------------
def _add_pairwise_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise ratios between key combine metrics."""
    key_metrics = ["Sprint_40yd", "Vertical_Jump", "Broad_Jump", "Bench_Press_Reps", "Weight", "Height"]
    
    for i, a in enumerate(key_metrics):
        for b in key_metrics[i + 1:]:
            df[f"{a}_div_{b}"] = _safe_div(df[a], df[b])
            df[f"{a}_mul_{b}"] = df[a] * df[b]

    return df


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------
def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    include_percentiles: bool = True,
    include_relative: bool = True,
    include_pairwise: bool = True,
    include_frequency: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the full feature matrix from raw train/test data.
    
    Target encoding is NOT done here — it must be done inside CV folds.
    This returns the base features that don't require the target.
    
    Returns:
        X_train, X_test: Feature DataFrames (no Id, no Drafted)
    """
    df = pd.concat(
        [train.drop(columns=["Drafted"]), test], axis=0
    ).reset_index(drop=True)
    n_train = len(train)

    # Fill categorical NAs
    for col in CAT_COLS:
        df[col] = df[col].fillna("Unknown").astype(str)

    # 1. Missingness features (BEFORE imputation)
    df = _add_missingness_features(df)

    # 2. Imputation
    df = _impute(df)

    # 3. Core athletic features
    df = _add_core_athletic(df)

    # 4. Percentile features
    if include_percentiles:
        df = _add_percentile_features(df)

    # 5. Relative features (z-scores, diffs)
    if include_relative:
        df = _add_relative_features(df)

    # 6. Frequency features
    if include_frequency:
        df = _add_frequency_features(df)

    # 7. Pairwise interactions
    if include_pairwise:
        df = _add_pairwise_interactions(df)

    # Clean up infinities and NaNs
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Split back
    X_train = df.iloc[:n_train].drop(columns=["Id"]).reset_index(drop=True)
    X_test = df.iloc[n_train:].drop(columns=["Id"]).reset_index(drop=True)

    return X_train, X_test


def get_cat_feature_indices(X: pd.DataFrame) -> list[int]:
    """Get column indices for categorical columns (for CatBoost)."""
    return [i for i, c in enumerate(X.columns) if c in CAT_COLS]


def get_numeric_only(X: pd.DataFrame) -> pd.DataFrame:
    """Drop categorical columns for models that need numeric-only input."""
    return X.drop(columns=[c for c in CAT_COLS if c in X.columns])

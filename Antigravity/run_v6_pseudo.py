"""
Antigravity v6 — Semi-Supervised Pseudo-Labeling Hybrid
Uses highly confident predictions from the 0.84595 model to augment the training data.
Retrains both CatBoost and the Neural Network on the augmented data.
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")

device = torch.device("cpu")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
SUB_DIR = Path(__file__).parent / "submissions_v6"
SUB_DIR.mkdir(parents=True, exist_ok=True)

CAT_COLS = ["School", "Player_Type", "Position_Type", "Position"]
PHYSICAL = [
    "Height", "Weight", "Sprint_40yd", "Vertical_Jump",
    "Bench_Press_Reps", "Broad_Jump", "Agility_3cone", "Shuttle", "Age",
]

# CatBoost Best Params (from user's Optuna)
CB_PARAMS = {
    'iterations': 1183,
    'learning_rate': 0.010492928203031996,
    'depth': 4,
    'l2_leaf_reg': 4.5594175959311425,
    'min_data_in_leaf': 7,
    'random_seed': 42,
    'eval_metric': 'Logloss', # Keep logloss for stable probabilities
    'verbose': 0
}

# ===================================================================
# 1. Feature Engineering
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

    # For CatBoost: Raw categories
    df_cb = df.copy()
    for col in CAT_COLS:
        df_cb[col] = df_cb[col].fillna("Unknown").astype(str)
        
    X_train_cb = df_cb.iloc[: len(train_df)].reset_index(drop=True)
    X_test_cb = df_cb.iloc[len(train_df) :].reset_index(drop=True)

    # For NN: Factorized and Scaled
    df_nn = df.copy()
    cat_dims = {}
    for col in CAT_COLS:
        df_nn[col] = df_nn[col].fillna("Unknown").astype(str)
        df_nn[col], uniques = pd.factorize(df_nn[col])
        cat_dims[col] = len(uniques)

    cont_cols = [c for c in df_nn.columns if c not in CAT_COLS + ["Id"]]
    scaler = StandardScaler()
    df_nn[cont_cols] = scaler.fit_transform(df_nn[cont_cols])
    df_nn[cont_cols] = df_nn[cont_cols].fillna(0)

    X_train_nn = df_nn.iloc[: len(train_df)].reset_index(drop=True)
    X_test_nn = df_nn.iloc[len(train_df) :].reset_index(drop=True)
    
    return X_train_cb, X_test_cb, X_train_nn, X_test_nn, cont_cols, cat_dims

# ===================================================================
# 2. Pseudo Label Logic
# ===================================================================
def get_pseudo_indices(test_probs, y, pos_threshold=0.90, neg_threshold=0.10):
    pos_candidates = np.flatnonzero(test_probs >= pos_threshold)
    neg_candidates = np.flatnonzero(test_probs <= neg_threshold)

    available_pos = len(pos_candidates)
    available_neg = len(neg_candidates)
    pos_rate = float(y.mean())
    neg_rate = 1.0 - pos_rate

    print(f"\\n[Pseudo-Label Bottleneck]")
    print(f"  Available > {pos_threshold:.2f}: {available_pos}")
    print(f"  Available < {neg_threshold:.2f}: {available_neg}")

    if available_pos == 0 or available_neg == 0:
        return np.array([], dtype=int), pd.Series([], dtype=int)

    max_total = int(min(available_pos / pos_rate, available_neg / neg_rate))
    capped_pos = min(available_pos, int(round(max_total * pos_rate)))
    capped_neg = min(available_neg, max_total - capped_pos)

    if max_total >= 2:
        capped_pos = max(1, capped_pos)
        capped_neg = max(1, capped_neg)

    while capped_pos > available_pos: capped_pos -= 1
    while capped_neg > available_neg: capped_neg -= 1

    selected_pos_idx = pos_candidates[np.argsort(-test_probs[pos_candidates])[:capped_pos]]
    selected_neg_idx = neg_candidates[np.argsort(test_probs[neg_candidates])[:capped_neg]]
    
    selected_idx = np.concatenate([selected_pos_idx, selected_neg_idx])
    selected_labels = np.concatenate([np.ones(capped_pos), np.zeros(capped_neg)])
    
    print(f"  Selected: {capped_pos} positive, {capped_neg} negative (Total: {len(selected_idx)})")
    return selected_idx, pd.Series(selected_labels, index=selected_idx, dtype=int)


# ===================================================================
# 3. Models
# ===================================================================
class TabularDataset(Dataset):
    def __init__(self, df, cont_cols, cat_cols, y=None):
        self.cont = torch.tensor(df[cont_cols].values, dtype=torch.float32)
        self.cat = torch.tensor(df[cat_cols].values, dtype=torch.long)
        self.y = torch.tensor(y.values, dtype=torch.float32) if y is not None else None

    def __len__(self): return len(self.cont)

    def __getitem__(self, idx):
        if self.y is not None: return self.cont[idx], self.cat[idx], self.y[idx]
        return self.cont[idx], self.cat[idx]

class TabularNN(nn.Module):
    def __init__(self, cat_dims, num_cont, hidden_dims=[256, 128]):
        super().__init__()
        self.embeds = nn.ModuleList([
            nn.Embedding(num_classes, min(50, (num_classes + 1) // 2))
            for col, num_classes in cat_dims.items()
        ])
        in_dim = sum(e.embedding_dim for e in self.embeds) + num_cont
        layers = []
        for out_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.GELU(), nn.Dropout(0.3)])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x_cont, x_cat):
        emb = torch.cat([e(x_cat[:, i]) for i, e in enumerate(self.embeds)], dim=1)
        return self.mlp(torch.cat([x_cont, emb], dim=1)).squeeze(1)


def train_catboost_pseudo(X, X_test, y, pseudo_X, pseudo_y, n_folds=10, seed=42):
    print(f"\n[CatBoost] Training {n_folds} folds with Pseudo-Labels...")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        X_tr, y_tr = X.iloc[tr_idx].copy(), y.iloc[tr_idx].copy()
        X_va, y_va = X.iloc[va_idx].copy(), y.iloc[va_idx].copy()
        
        # Inject Pseudo-Labels safely into train only
        if len(pseudo_X) > 0:
            X_tr = pd.concat([X_tr, pseudo_X], axis=0).reset_index(drop=True)
            y_tr = pd.concat([y_tr, pseudo_y], axis=0).reset_index(drop=True)
            
        tp = Pool(X_tr.drop(columns=["Id"]), y_tr, cat_features=CAT_COLS)
        vp = Pool(X_va.drop(columns=["Id"]), y_va, cat_features=CAT_COLS)
        
        p = CB_PARAMS.copy()
        p["random_seed"] = seed + fold
        model = CatBoostClassifier(**p)
        model.fit(tp, eval_set=vp, early_stopping_rounds=100)
        
        oof[va_idx] = model.predict_proba(vp)[:, 1]
        test_pred += model.predict_proba(Pool(X_test.drop(columns=["Id"]), cat_features=CAT_COLS))[:, 1] / n_folds
        
    print(f">> CatBoost Pseudo OOF AUC: {roc_auc_score(y, oof):.5f}")
    return oof, test_pred

def train_nn_pseudo(X, X_test, y, pseudo_X, pseudo_y, cont_cols, cat_dims, n_folds=10, seed=42):
    print(f"\n[Neural Network] Training {n_folds} folds with Pseudo-Labels...")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    
    test_dl = DataLoader(TabularDataset(X_test, cont_cols, CAT_COLS), batch_size=256, shuffle=False)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        X_tr, y_tr = X.iloc[tr_idx].copy(), y.iloc[tr_idx].copy()
        X_va, y_va = X.iloc[va_idx].copy(), y.iloc[va_idx].copy()
        
        if len(pseudo_X) > 0:
            X_tr = pd.concat([X_tr, pseudo_X], axis=0).reset_index(drop=True)
            y_tr = pd.concat([y_tr, pseudo_y], axis=0).reset_index(drop=True)
            
        tr_dl = DataLoader(TabularDataset(X_tr, cont_cols, CAT_COLS, y_tr), batch_size=128, shuffle=True)
        va_dl = DataLoader(TabularDataset(X_va, cont_cols, CAT_COLS, y_va), batch_size=256, shuffle=False)
        
        model = TabularNN(cat_dims, len(cont_cols)).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        
        best_va_auc, best_va_preds, best_test_preds, patience = 0, None, None, 0
        for epoch in range(50):
            model.train()
            for x_cont, x_cat, targets in tr_dl:
                x_cont, x_cat, targets = x_cont.to(device), x_cat.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_cont, x_cat), targets)
                loss.backward()
                optimizer.step()
                
            model.eval()
            va_preds, va_targets = [], []
            with torch.no_grad():
                for x_cont, x_cat, targets in va_dl:
                    va_preds.extend(torch.sigmoid(model(x_cont.to(device), x_cat.to(device))).cpu().numpy())
                    va_targets.extend(targets.numpy())
            
            auc = roc_auc_score(va_targets, va_preds)
            if auc > best_va_auc:
                best_va_auc, best_va_preds, patience = auc, va_preds, 0
                best_test_preds = []
                with torch.no_grad():
                    for x_cont, x_cat in test_dl:
                        best_test_preds.extend(torch.sigmoid(model(x_cont.to(device), x_cat.to(device))).cpu().numpy())
            else:
                patience += 1
            if patience >= 10: break
                
        oof[va_idx] = best_va_preds
        test_pred += np.array(best_test_preds) / n_folds

    print(f">> Neural Network Pseudo OOF AUC: {roc_auc_score(y, oof):.5f}")
    return oof, test_pred

# ===================================================================
# 4. Main Execution
# ===================================================================
def rank01(v):
    return pd.Series(v).rank(pct=True, method="average").values

def main():
    print("=" * 60)
    print("ANTIGRAVITY v6 - Pseudo-Label Hybrid")
    print("=" * 60)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    y = train["Drafted"].reset_index(drop=True)
    test_ids = test["Id"].reset_index(drop=True)

    X_train_cb, X_test_cb, X_train_nn, X_test_nn, cont_cols, cat_dims = build_features(train, test)
    
    print("\n[Loading 0.84595 Model for Pseudo-Labels]")
    try:
        hybrid_path = ROOT / "Antigravity" / "submissions_v5" / "v5_hybrid_nn_catboost.csv"
        prev_test_preds = pd.read_csv(hybrid_path)["Drafted"].values
    except FileNotFoundError:
        print("Error: Could not find v5_hybrid_nn_catboost.csv. Exiting.")
        return

    ps_idx, ps_y = get_pseudo_indices(prev_test_preds, y)
    
    ps_X_cb = X_test_cb.iloc[ps_idx].copy() if len(ps_idx) > 0 else X_test_cb.iloc[[]]
    ps_X_nn = X_test_nn.iloc[ps_idx].copy() if len(ps_idx) > 0 else X_test_nn.iloc[[]]

    oof_cb, test_cb = train_catboost_pseudo(X_train_cb, X_test_cb, y, ps_X_cb, ps_y)
    oof_nn, test_nn = train_nn_pseudo(X_train_nn, X_test_nn, y, ps_X_nn, ps_y, cont_cols, cat_dims)
    
    print("\n[Final Ensembling]")
    blend_test = (rank01(test_nn) + rank01(test_cb)) / 2
    
    path = SUB_DIR / "v6_pseudo_hybrid.csv"
    pd.DataFrame({"Id": test_ids, "Drafted": blend_test}).to_csv(path, index=False)
    print(f"SUCCESS! Pseudo Hybrid saved to: {path}")
    
    pd.DataFrame({"Id": test_ids, "Drafted": test_cb}).to_csv(SUB_DIR / "v6_pseudo_catboost.csv", index=False)
    pd.DataFrame({"Id": test_ids, "Drafted": test_nn}).to_csv(SUB_DIR / "v6_pseudo_nn.csv", index=False)

if __name__ == "__main__":
    main()

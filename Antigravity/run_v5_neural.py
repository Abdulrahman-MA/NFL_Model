"""
Antigravity v5 — Neural Network (PyTorch) + CatBoost Hybrid
Uses Entity Embeddings for categorical variables and an MLP for continuous variables.
Blends with the LB-proven CatBoost baseline.
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

warnings.filterwarnings("ignore")

# Force CPU for stability on small datasets
device = torch.device("cpu")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "Data"
SUB_DIR = Path(__file__).parent / "submissions_v5"
SUB_DIR.mkdir(parents=True, exist_ok=True)

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

    # Missing flags
    for col in PHYSICAL:
        if df[col].isnull().sum() > 0:
            df[f"{col}_Is_Missing"] = df[col].isnull().astype(int)

    # Imputation
    for col in PHYSICAL:
        df[col] = df[col].fillna(df.groupby("Position")[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

    # Derived Physics
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

    # Clean categoricals
    cat_dims = {}
    for col in CAT_COLS:
        df[col] = df[col].fillna("Unknown").astype(str)
        # Factorize to continuous integers 0 to N
        df[col], uniques = pd.factorize(df[col])
        cat_dims[col] = len(uniques)

    cont_cols = [c for c in df.columns if c not in CAT_COLS + ["Id"]]
    
    # Scale continuous
    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])
    df[cont_cols] = df[cont_cols].fillna(0) # Safety

    X_train = df.iloc[: len(train_df)].reset_index(drop=True)
    X_test = df.iloc[len(train_df) :].reset_index(drop=True)
    
    return X_train, X_test, cont_cols, cat_dims

# ===================================================================
# 2. PyTorch Dataset & Model
# ===================================================================
class TabularDataset(Dataset):
    def __init__(self, df, cont_cols, cat_cols, y=None):
        self.cont = torch.tensor(df[cont_cols].values, dtype=torch.float32)
        self.cat = torch.tensor(df[cat_cols].values, dtype=torch.long)
        self.y = torch.tensor(y.values, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.cont)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.cont[idx], self.cat[idx], self.y[idx]
        return self.cont[idx], self.cat[idx]

class TabularNN(nn.Module):
    def __init__(self, cat_dims, num_cont, hidden_dims=[256, 128]):
        super().__init__()
        
        # Setup embeddings
        self.embeds = nn.ModuleList([
            nn.Embedding(num_classes, min(50, (num_classes + 1) // 2))
            for col, num_classes in cat_dims.items()
        ])
        
        total_embed_dim = sum(e.embedding_dim for e in self.embeds)
        in_dim = total_embed_dim + num_cont
        
        layers = []
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.3))
            in_dim = out_dim
            
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x_cont, x_cat):
        # Pass categorical features through embeddings
        embed_out = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
        x_cat_emb = torch.cat(embed_out, dim=1)
        
        # Concatenate with continuous
        x = torch.cat([x_cont, x_cat_emb], dim=1)
        return self.mlp(x).squeeze(1)

# ===================================================================
# 3. Training Loop
# ===================================================================
def train_nn(X, X_test, y, cont_cols, cat_dims, n_folds=10, seed=42):
    print(f"\n[Neural Network] Training {n_folds} folds...")
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    
    # Test DataLoader
    test_ds = TabularDataset(X_test, cont_cols, CAT_COLS)
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        # DataLoaders
        tr_ds = TabularDataset(X.iloc[tr_idx], cont_cols, CAT_COLS, y.iloc[tr_idx])
        va_ds = TabularDataset(X.iloc[va_idx], cont_cols, CAT_COLS, y.iloc[va_idx])
        
        tr_dl = DataLoader(tr_ds, batch_size=128, shuffle=True)
        va_dl = DataLoader(va_ds, batch_size=256, shuffle=False)
        
        model = TabularNN(cat_dims, len(cont_cols)).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        
        best_va_auc = 0
        best_va_preds = None
        best_test_preds = None
        patience = 0
        
        for epoch in range(50): # Max epochs
            model.train()
            for x_cont, x_cat, targets in tr_dl:
                x_cont, x_cat, targets = x_cont.to(device), x_cat.to(device), targets.to(device)
                optimizer.zero_grad()
                out = model(x_cont, x_cat)
                loss = criterion(out, targets)
                loss.backward()
                optimizer.step()
                
            # Eval
            model.eval()
            va_preds = []
            va_targets = []
            with torch.no_grad():
                for x_cont, x_cat, targets in va_dl:
                    x_cont, x_cat = x_cont.to(device), x_cat.to(device)
                    preds = torch.sigmoid(model(x_cont, x_cat)).cpu().numpy()
                    va_preds.extend(preds)
                    va_targets.extend(targets.numpy())
                    
            auc = roc_auc_score(va_targets, va_preds)
            
            if auc > best_va_auc:
                best_va_auc = auc
                best_va_preds = va_preds
                patience = 0
                
                # Predict on test
                te_preds = []
                with torch.no_grad():
                    for x_cont, x_cat in test_dl:
                        x_cont, x_cat = x_cont.to(device), x_cat.to(device)
                        preds = torch.sigmoid(model(x_cont, x_cat)).cpu().numpy()
                        te_preds.extend(preds)
                best_test_preds = te_preds
            else:
                patience += 1
                
            if patience >= 10: # Early stopping
                break
                
        print(f"  Fold {fold} AUC: {best_va_auc:.5f} (Stopped at epoch {epoch+1})")
        oof[va_idx] = best_va_preds
        test_pred += np.array(best_test_preds) / n_folds

    total_auc = roc_auc_score(y, oof)
    print(f">> Neural Network OOF AUC: {total_auc:.5f}")
    return oof, test_pred

# ===================================================================
# 4. Main Execution
# ===================================================================
def rank01(v):
    return pd.Series(v).rank(pct=True, method="average").values

def main():
    print("=" * 60)
    print("ANTIGRAVITY v5 - Neural Network Hybrid")
    print("=" * 60)

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    y = train["Drafted"].reset_index(drop=True)
    test_ids = test["Id"].reset_index(drop=True)

    X_train, X_test, cont_cols, cat_dims = build_features(train, test)
    
    # 1. Train Neural Network
    oof_nn, test_nn = train_nn(X_train, X_test, y, cont_cols, cat_dims)
    
    # 2. Load CatBoost Predictions
    # We will load the proven 10-seed predictions from V3
    print("\n[Hybrid Blending] Loading CatBoost 10-seed predictions...")
    try:
        cb_test_df = pd.read_csv(ROOT / "Antigravity" / "submissions_v3" / "old_36f_10seeds.csv")
        test_cb = cb_test_df["Drafted"].values
        # We don't have the exact OOF saved in a CSV for CatBoost, so we will just rank blend the test preds
        blend_test = (rank01(test_nn) + rank01(test_cb)) / 2
        
        # Save Hybrid
        path = SUB_DIR / "v5_hybrid_nn_catboost.csv"
        pd.DataFrame({"Id": test_ids, "Drafted": blend_test}).to_csv(path, index=False)
        print(f"Hybrid Blend saved to: {path}")
        
    except FileNotFoundError:
        print("Warning: Could not find V3 CatBoost predictions to blend with. Generating standalone NN only.")
        
    # Save standalone NN
    path_nn = SUB_DIR / "v5_neural_standalone.csv"
    pd.DataFrame({"Id": test_ids, "Drafted": test_nn}).to_csv(path_nn, index=False)
    print(f"Standalone NN saved to: {path_nn}")

if __name__ == "__main__":
    main()

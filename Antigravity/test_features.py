"""Quick test of feature building."""
import sys
sys.path.insert(0, "Antigravity")
import pandas as pd
from features import build_features

train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")
X_train, X_test = build_features(train, test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Total features: {X_train.shape[1]}")
print(f"Nulls: {X_train.isnull().sum().sum()}")
print(f"Sample columns (first 40):")
for c in list(X_train.columns)[:40]:
    print(f"  {c}")
print("...")
print(f"Total columns: {X_train.shape[1]}")

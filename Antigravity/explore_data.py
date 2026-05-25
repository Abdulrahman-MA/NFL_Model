"""Quick data exploration script."""
import pandas as pd
import numpy as np

train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')

print('=== TRAIN SHAPE ===')
print(train.shape)
print('\n=== TEST SHAPE ===')
print(test.shape)

print('\n=== COLUMNS ===')
print(list(train.columns))

print('\n=== DTYPES ===')
print(train.dtypes)

print('\n=== DESCRIBE (numeric) ===')
print(train.describe().to_string())

print('\n=== HEAD ===')
print(train.head(5).to_string())

print('\n=== TARGET DISTRIBUTION ===')
print(train['Drafted'].value_counts())
print(f"Draft rate: {train['Drafted'].mean():.4f}")

print('\n=== NULL COUNTS (TRAIN) ===')
print(train.isnull().sum())

print('\n=== NULL COUNTS (TEST) ===')
print(test.isnull().sum())

print('\n=== NULL PERCENTAGES (TRAIN) ===')
for col in train.columns:
    pct = train[col].isnull().mean() * 100
    if pct > 0:
        print(f"  {col}: {pct:.1f}%")

print('\n=== UNIQUE CATEGORICAL VALUES ===')
for c in ['School', 'Player_Type', 'Position_Type', 'Position']:
    print(f"  {c}: {train[c].nunique()} unique (train), {test[c].nunique()} unique (test)")
    # Check for test-only values
    test_only = set(test[c].dropna().unique()) - set(train[c].dropna().unique())
    if test_only:
        print(f"    TEST-ONLY values: {test_only}")

print('\n=== Year distribution (TRAIN) ===')
print(train['Year'].value_counts().sort_index())

print('\n=== Year distribution (TEST) ===')
print(test['Year'].value_counts().sort_index())

print('\n=== Position draft rates ===')
pos_draft = train.groupby('Position')['Drafted'].agg(['mean', 'count']).sort_values('mean', ascending=False)
print(pos_draft.to_string())

print('\n=== Position_Type draft rates ===')
pt_draft = train.groupby('Position_Type')['Drafted'].agg(['mean', 'count']).sort_values('mean', ascending=False)
print(pt_draft.to_string())

print('\n=== Player_Type draft rates ===')
plt_draft = train.groupby('Player_Type')['Drafted'].agg(['mean', 'count']).sort_values('mean', ascending=False)
print(plt_draft.to_string())

print('\n=== School top-20 by count ===')
school_stats = train.groupby('School')['Drafted'].agg(['mean', 'count']).sort_values('count', ascending=False).head(20)
print(school_stats.to_string())

print('\n=== Correlation with target (numeric) ===')
num_cols = train.select_dtypes(include=[np.number]).columns.drop('Drafted')
corrs = train[num_cols].corrwith(train['Drafted']).sort_values(ascending=False)
print(corrs.to_string())

print('\n=== Sample submission ===')
sample = pd.read_csv('Data/sample_submission.csv')
print(sample.head())
print(sample.describe())

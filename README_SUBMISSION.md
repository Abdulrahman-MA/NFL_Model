# NFL Draft Prediction - Clean Legal Submission

This root folder is the audit-ready version of the project.

## Show These Files

- `NFL_Clean_Legal_Submission.ipynb`
- `legal_tournament/run_legal_tournament.py`
- `legal_tournament/results/tournament_summary.csv`
- `Data/train.csv`
- `Data/test.csv`
- `Data/sample_submission.csv`
- `Data/submission.csv`

## Notebook Cells To Show

In `NFL_Clean_Legal_Submission.ipynb`:

- Cell 0: short description that the pipeline uses only provided competition data.
- Cell 1: imports and loading `Data/train.csv` / `Data/test.csv`.
- Cell 2: legal feature engineering and CatBoost matrix creation.
- Cell 3: 10-fold CatBoost validation and test-prediction ensembling.
- Cell 4: final submission export to `Data/submission.csv`.

## Current Model

The active submission is currently `Data/submission.csv`.

It was produced by `legal_tournament/run_legal_tournament.py` from the best local-OOF tournament candidate, `blend_rank_random_search`.

The clean notebook remains a simpler audit baseline: it creates a 10-fold CatBoost ensemble where each fold trains on 90% of the provided training rows, validates on the remaining 10%, predicts the test set, and averages the final test probabilities across folds.

## Legal Tournament

The `legal_tournament/` folder contains the legal leaderboard-chasing framework. It trains several model families, writes candidate submissions to `legal_tournament/submissions/`, and records OOF ROC-AUC results in `legal_tournament/results/tournament_summary.csv`.

The current active `Data/submission.csv` was copied from the best local-OOF tournament candidate.

## Archived Trials

The `trials/` folder contains older notebooks, experimental submissions, and generated CatBoost logs. These are preserved for traceability but are not part of the clean submission path.

The clean notebook does not use external source reconstruction data.

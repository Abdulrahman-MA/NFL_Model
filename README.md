# NFL Draft Prediction

Predict whether a college player gets drafted into the NFL from their NFL Combine measurements, position and school, using gradient-boosted trees and blended ensembles.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Task](https://img.shields.io/badge/task-binary%20classification-orange)
![Metric](https://img.shields.io/badge/metric-ROC--AUC-green)

## Overview

This is a binary classification task: given a prospect's combine results, estimate the probability that they were drafted (`Drafted = 1`).

- **Metric:** ROC-AUC
- **Training data:** 2,781 players (about 64.8% drafted), combine years 2009–2019
- **Test data:** 696 players, same year range
- **Data policy:** every model uses only the provided `Data/train.csv` and `Data/test.csv`. No external data is used.

### Features in the raw data

| Group | Columns |
|-------|---------|
| Identity | `Id`, `Year`, `School` |
| Body | `Age`, `Height`, `Weight` |
| Combine drills | `Sprint_40yd`, `Vertical_Jump`, `Bench_Press_Reps`, `Broad_Jump`, `Agility_3cone`, `Shuttle` |
| Role | `Player_Type`, `Position_Type`, `Position` |
| Target | `Drafted` (train only) |

## Approach

1. **Missing-value handling.** Add `*_Is_Missing` flags, then impute with the position median. Whether a player skipped a drill is itself informative.
2. **Feature engineering.**
   - BMI, a speed score, an explosiveness index (vertical × broad jump) and a power score.
   - Each metric's difference from its position average.
   - School frequency counts.
3. **Models.** CatBoost, LightGBM, XGBoost, Random Forest and Extra Trees, each trained with 10-fold stratified CV.
4. **Ensembling.** Rank-averaging, probability averaging, a logistic-regression meta-model, and a random-search blend of rank-normalised predictions.

## Results

Scores are local cross-validated ROC-AUC. Full results are in [`legal_tournament/results/tournament_summary.csv`](legal_tournament/results/tournament_summary.csv).

| Model | Pooled OOF AUC | Mean fold AUC |
|-------|---------------:|--------------:|
| **Rank blend (random-search weights)** | **0.8428** | – |
| Rank blend (top 4) | 0.8392 | – |
| Extra Trees | 0.8388 | 0.8396 |
| Probability mean (all models) | 0.8386 | – |
| Random Forest | 0.8332 | 0.8331 |
| CatBoost (rich features) | 0.8052 | 0.8391 |
| LightGBM (rich features) | 0.8048 | 0.8369 |
| XGBoost (rich features) | 0.8055 | 0.8357 |

> [!NOTE]
> Pooled OOF AUC concatenates all out-of-fold predictions before scoring. Mean fold AUC scores each fold separately. For the boosted models the two differ noticeably (about 0.80 vs 0.84). The likely cause is that each fold's predictions are not on the same scale, which hurts the pooled score. This is the reason the final blend uses rank-normalised predictions. The blend weights were tuned on the same OOF predictions, so 0.8428 is somewhat optimistic.

## Repository structure

```
NFL_Model/
├── Data/                              # train.csv, test.csv, sample_submission.csv, submission.csv
├── NFL_Clean_Legal_Submission.ipynb   # Simple, readable CatBoost 10-fold baseline
├── NFL_Final_Submission.ipynb         # Earlier final-submission notebook
├── legal_tournament/                  # Multi-model comparison + blending (produces the final submission)
│   ├── run_legal_tournament.py
│   ├── results/                       # Scores per model / blend
│   └── submissions/                   # Candidate prediction files
├── Antigravity/                       # Earlier iterations (v1–v7): tuning, stacking, neural nets, pseudo-labeling, AutoGluon
├── trials/                            # Archived early notebooks and submissions
├── requirements.txt
└── README_SUBMISSION.md               # Notes for the submission audit
```

## Getting started

```bash
git clone https://github.com/Abdulrahman-MA/NFL_Model.git
cd NFL_Model

python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

### Run the baseline notebook

```bash
jupyter lab NFL_Clean_Legal_Submission.ipynb
```

It runs the CatBoost 10-fold pipeline and writes `Data/submission.csv`.

### Run the full model comparison

```bash
python legal_tournament/run_legal_tournament.py
```

This trains all model families and writes candidate files to `legal_tournament/submissions/` and scores to `legal_tournament/results/`. It also copies the best candidate to `Data/submission.csv`, **overwriting the existing file**.

The scripts in `Antigravity/` are earlier experiments. `run_v7_autogluon.py` additionally requires `autogluon`, and `run_v5_neural.py` requires `torch`.

## Tech stack

Python · pandas · NumPy · scikit-learn · CatBoost · LightGBM · XGBoost · category_encoders · Optuna · (experiments: PyTorch, AutoGluon)

## Possible improvements

- Use nested CV (or a held-out split) so the blend weights are not evaluated on the data they were tuned on.
- Calibrate the fold models so pooled and per-fold AUC agree.
- Add position-aware features such as percentile rank within position and year.

## License

No license has been specified yet. Add a `LICENSE` file to state how others may use this code. The dataset comes from the competition it was provided for, so check its terms before reusing it.

# Legal Model Tournament

This folder is for leaderboard-chasing experiments that still use only the competition-provided files:

- `../Data/train.csv`
- `../Data/test.csv`
- `../Data/sample_submission.csv`

The script writes candidate submissions to `submissions/` and model scores to `results/`.

Run:

```powershell
.\.venv\Scripts\python.exe legal_tournament\run_legal_tournament.py
```

The script also copies the best local-OOF candidate to `../Data/submission.csv`.

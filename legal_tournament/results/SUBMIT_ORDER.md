# Suggested Legal Submission Order

Use the leaderboard sparingly. These candidates are all generated from `Data/train.csv` and `Data/test.csv` only.

1. `Data/submission.csv`
   - Same as `legal_tournament/submissions/blend_rank_random_search.csv`
   - Best local OOF in the current tournament: `0.842810`

2. `legal_tournament/submissions/extra_trees_rich_10fold.csv`
   - Best single-model tournament candidate: `0.838795`
   - More stable pooled OOF than the CatBoost CV variants.

3. `legal_tournament/submissions/blend_rank_mean_top4.csv`
   - Conservative rank blend of the top four local candidates: `0.839153`

4. `legal_tournament/submissions/blend_prob_mean_all.csv`
   - Probability average across all model families: `0.838631`

Record each public leaderboard score before submitting the next file. The next tuning step should use those scores to decide whether the public split prefers rank blends, tree bagging, or smoother probability averages.

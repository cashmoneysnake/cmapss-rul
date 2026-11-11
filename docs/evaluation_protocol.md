# Evaluation protocol (locked)

Written before any training. Each engine produces many overlapping windows. If window 40 of engine #7 is in training and window 41 of engine #7 is in validation, the model has effectively seen the answer and the score looks great for the wrong reason. Splitting by whole engine fixes that. Tuning on the official test set has the same flaw: you end up picking whatever happens to score well on it.

## Rules
1. **Split by engine ID.** Validation is a held-out set of whole engines (GroupKFold by unit number, or a fixed held-out set). Never split windows randomly.
2. **Tune on validation only.** Model choice, hyperparameters, window length, and epochs are all chosen on validation.
3. **Touch the official test set once per final model.** No iterating against it.
4. **Seeds.** Neural nets: at least 3 seeds, report mean and standard deviation. XGBoost: report the same if it has any randomness.
5. **RUL target cap: 125 cycles** for training targets (RUL cap is also an ablation).
6. **Window length: 30 cycles by default** (window length is an ablation).
7. **Test scoring: last window of each test engine** compared against NASA's provided true-RUL file for that subset.
8. **Features.** Drop zero-variance sensors. For FD002 and FD004, normalize per operating condition (assign each cycle to its operating-condition cluster, then scale within that cluster). Scaling statistics are fit on training data only and reused for validation and test.
9. **Logging.** Every run goes to MLflow with subset, model, seed, window, cap, and metrics. Only logged runs appear in the results table.

## Metrics
- **RMSE** on the last-window predictions.
- **NASA asymmetric score.** With d = predicted RUL minus true RUL for each test engine:
  - if d < 0 (early prediction): exp(-d / 13) - 1
  - if d >= 0 (late prediction): exp(d / 10) - 1
  - sum over engines. Lower is better. Late predictions are punished more than early ones because predicting too much remaining life is the dangerous error.

## Subset facts
| Subset | Operating conditions | Fault modes |
|---|---|---|
| FD001 | 1 | 1 |
| FD002 | 6 | 1 |
| FD003 | 1 | 2 |
| FD004 | 6 | 2 |

## Honesty notes
- Report where the deep model did not beat XGBoost, with a short explanation.
- Compare to published numbers only like-for-like (same subset, same cap, last-window test scoring). Everything else is labeled "not directly comparable".

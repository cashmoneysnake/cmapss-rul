# Literature comparison

Goal: check my results against published C-MAPSS work, like-for-like only, and say plainly where a comparison is not fair.

**Scope of this search.** I read two papers in full detail and a third only for its comparison table. The search was not exhaustive, so absence of a paper here does not mean it is not comparable. Anything not read in the original is marked as such.

## My setup (for matching)
RUL cap 125, window 30, 80/20 train/validation split by engine, per-condition scaling (KMeans, 6 clusters) fit on training data only, test scoring on the last window of each test engine against NASA's RUL file, 3 seeds, mean and sample standard deviation. Source: `docs/results.md` (generated from MLflow).

## Like-for-like table (closest match found)
Rule: same subset, RUL cap 125, one prediction per test engine. Partial setup details are flagged.

| Subset | Model (paper) | RMSE | NASA score | Setup match | Citation |
|---|---|---|---|---|---|
| FD001 | Proposed framework, "Ours (Avg)" / "Ours (Best)" | 17.25 / 16.22 | not reported | Cap 125, window 30, 80/20 split by engine ID, RMSE "computed on final RUL predictions per engine". NOT stated: number of runs behind "Avg"/"Best", how short test engines are handled, or which architecture produced the Table I rows. Preprint, not peer reviewed. | Sharma, K. "Uncertainty-Aware Deep Learning Framework for Remaining Useful Life Prediction in Turbofan Engines with Learned Aleatoric Uncertainty." arXiv:2511.19124v2, Nov 2025, Table I. https://arxiv.org/abs/2511.19124 |
| FD004 | same | 21.91 / 19.98 | not reported | same as above (per-condition scaling with KMeans k=6, as mine) | same |

## Not directly comparable
| Paper | What differs | Citation |
|---|---|---|
| Proposed TDDN model: FD001 RMSE 9.47, score 214.17; FD004 RMSE 11.16, score 997.69 | RUL cap 120 (mine 125), window 64 (mine 30). The paper does not say whether it scores only the last window of each test engine (it only says the score sums over "the total number of samples in the test dataset"). | Qin, Y., Cai, N., Gao, C., Zhang, Y., Cheng, Y., Chen, X. "Remaining Useful Life Prediction Using Temporal Deep Degradation Network for Complex Machinery with Attention-based Feature Extraction." arXiv:2202.10916, Feb 2022, Table 4. https://arxiv.org/abs/2202.10916 |
| LSTM baseline (as reported in the TDDN paper's Table 4, citing its reference [21]): FD001 RMSE 16.14, score 338; FD004 RMSE 28.17, score 5550 | Second-hand: I read these in the TDDN paper's table, not in the original paper, and I have not checked the original's cap, window, or test protocol. Do not use as a like-for-like row until the original is read. | Same TDDN paper, Table 4 (row "LSTM[21]") |
| Other cited baselines in both papers above (CNN, Bi-LSTM, TCN, and so on) | Not read in their original papers; setups unknown. | n/a |

## My results vs published
Mine (test set, last window per engine, mean +- sd over 3 seeds; from `docs/results.md`):

| Subset | Model | RMSE |
|---|---|---|
| FD001 | GRU, window 30 | 15.28 +- 0.13 |
| FD001 | XGBoost, window 30 | 16.22 +- 0.35 |
| FD004 | GRU, window 70 (tuned on validation) | 25.94 +- 0.29 |
| FD004 | GRU, window 30 | 26.88 +- 0.21 |

- **FD001:** my RMSE (15.28 for GRU, 16.22 for XGBoost) is below the like-for-like paper's average (17.25) and its best run (16.22) for the GRU, and equal to its best run for XGBoost. That paper does not say how many runs "Avg" and "Best" cover, and it reports a single best run, so my 3-seed mean is the more conservative number.
- **FD004:** my RMSE (about 26 to 27) is clearly worse than the like-for-like paper's (21.91 average, 19.98 best). I have not diagnosed why. Untested possibilities: a much larger architecture in that paper (Inception blocks, Bi-LSTM, attention), differences in how short or unusual test engines are handled (11 of 248 FD004 test engines have fewer than 30 cycles, and 20 have fewer than 50), and unstated protocol details. I do not claim any of these is the cause.
- Both papers' numbers are single-source and not independently reproduced here.

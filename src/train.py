"""Train one model and log the run to MLflow.

Examples (from the repo root):
    python -m src.train --subset FD001 --model xgboost --seed 0
    python -m src.train --subset FD001 --model xgboost --seed 0 --final   # also scores the test set
"""
import argparse
import os
from pathlib import Path

import mlflow
import numpy as np

from src.data.pipeline import build_datasets
from src.evaluate import nasa_score, rmse, score_test_set
from src.features import window_features

ROOT = Path(__file__).resolve().parents[1]


def fit_xgboost(data, seed, n_estimators):
    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=0.05, max_depth=5,
                         subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    model.fit(window_features(data.X_train), data.y_train)
    return lambda X: model.predict(window_features(X))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--model", default="xgboost", choices=["xgboost"])
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--cap", type=int, default=125)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--final", action="store_true",
                    help="also score the official test set (do this once per final model)")
    args = ap.parse_args()

    # default store: mlflow.db in the repo root (same one `mlflow ui` shows)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment("cmapss-rul")

    data = build_datasets(args.subset, args.window, args.cap, args.seed, args.val_frac)
    print(f"train windows {data.X_train.shape}, val windows {data.X_val.shape}")

    with mlflow.start_run(run_name=f"{args.model}-{args.subset}-seed{args.seed}"):
        mlflow.log_params({**vars(args), "n_features": len(data.feature_cols)})
        predict = fit_xgboost(data, args.seed, args.n_estimators)

        # validation: every window of the held-out engines (targets capped)
        y_pred = predict(data.X_val)
        metrics = {"val_rmse": rmse(data.y_val, y_pred),
                   "val_nasa_score": nasa_score(data.y_val, y_pred)}
        if args.final:
            test = score_test_set(predict, data.test_engines, data.test_rul, args.window)
            metrics.update({f"test_{k}": v for k, v in test.items()})
        mlflow.log_metrics(metrics)
        print({k: round(v, 2) for k, v in metrics.items()})


if __name__ == "__main__":
    main()

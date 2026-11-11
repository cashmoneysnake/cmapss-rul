"""Train one model and log the run to MLflow.

Examples (from the repo root):
    python -m src.train --subset FD001 --model xgboost --seed 0
    python -m src.train --subset FD001 --model xgboost --seed 0 --final   # also scores the test set
"""
import argparse
import json
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
    return (lambda X: model.predict(window_features(X))), None


def fit_nn(data, model_name, seed, epochs, hidden, lr, batch_size, cap):
    import torch
    from src.models.factory import build_model

    torch.manual_seed(seed)
    np.random.seed(seed)
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    X_tr, y_tr = to_t(data.X_train), to_t(data.y_train / cap)   # targets scaled to 0..1
    X_va = to_t(data.X_val)

    model = build_model(model_name, X_tr.shape[2], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    def predict(X):
        model.eval()
        with torch.no_grad():
            X = X if torch.is_tensor(X) else to_t(X)
            out = torch.cat([model(X[i:i + 2048]) for i in range(0, len(X), 2048)])
        return np.clip(out.numpy() * cap, 0, cap)

    best_rmse, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
        val_rmse = rmse(data.y_val, predict(X_va))
        mlflow.log_metric("val_rmse_by_epoch", val_rmse, step=epoch)
        print(f"epoch {epoch + 1:>3}/{epochs}  val_rmse {val_rmse:.2f}")
        if val_rmse < best_rmse:   # keep the best epoch, chosen on validation only
            best_rmse = val_rmse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict, model


def save_bundle(out: Path, model, data, args):
    """Save everything the API needs: weights, scaler statistics, and the settings used."""
    import torch
    if model is None:
        raise SystemExit("--save-dir only works for neural models (lstm, gru, cnn, transformer)")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "scaler.json").write_text(json.dumps(data.scaler.to_dict()))
    (out / "config.json").write_text(json.dumps({
        "model": args.model, "subset": args.subset, "window": args.window, "cap": args.cap,
        "hidden": args.hidden, "seed": args.seed, "n_features": len(data.feature_cols),
        "feature_cols": data.feature_cols}, indent=2))
    print(f"saved serving bundle to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--model", default="xgboost", choices=["xgboost", "lstm", "gru", "cnn", "transformer"])
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--cap", type=int, default=125)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=0,
                    help="seed for the engine train/val split (kept fixed across model seeds)")
    ap.add_argument("--n-estimators", type=int, default=300)   # xgboost
    ap.add_argument("--epochs", type=int, default=30)          # neural nets
    ap.add_argument("--hidden", type=int, default=64)          # neural nets
    ap.add_argument("--lr", type=float, default=1e-3)          # neural nets
    ap.add_argument("--batch-size", type=int, default=128)     # neural nets
    ap.add_argument("--val-min-cycle", type=int, default=0,
                    help="also report validation metrics (_common) on windows ending at or after this "
                         "cycle, so different window lengths are scored on the same targets")
    ap.add_argument("--save-dir", default="",
                    help="neural models only: save weights, scaler and config here for the API")
    ap.add_argument("--final", action="store_true",
                    help="also score the official test set (do this once per final model)")
    args = ap.parse_args()

    # default store: mlflow.db in the repo root (same one `mlflow ui` shows)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    mlflow.set_experiment("cmapss-rul")

    data = build_datasets(args.subset, args.window, args.cap, args.split_seed, args.val_frac)
    print(f"train windows {data.X_train.shape}, val windows {data.X_val.shape}")

    with mlflow.start_run(run_name=f"{args.model}-{args.subset}-w{args.window}-seed{args.seed}"):
        mlflow.log_params({**vars(args), "n_features": len(data.feature_cols)})
        if args.model == "xgboost":
            predict, model = fit_xgboost(data, args.seed, args.n_estimators)
        else:
            predict, model = fit_nn(data, args.model, args.seed, args.epochs, args.hidden,
                                    args.lr, args.batch_size, args.cap)

        # validation: every window of the held-out engines (targets capped)
        y_pred = predict(data.X_val)
        metrics = {"val_rmse": rmse(data.y_val, y_pred),
                   "val_nasa_score": nasa_score(data.y_val, y_pred)}
        if args.final:
            test = score_test_set(predict, data.test_engines, data.test_rul, args.window)
            metrics.update({f"test_{k}": v for k, v in test.items()})
        if args.val_min_cycle:
            keep = data.val_end_cycle >= args.val_min_cycle
            metrics["val_rmse_common"] = rmse(data.y_val[keep], y_pred[keep])
            metrics["val_nasa_score_common"] = nasa_score(data.y_val[keep], y_pred[keep])
            metrics["val_windows_common"] = float(keep.sum())
        if args.save_dir:
            save_bundle(Path(args.save_dir), model, data, args)
        mlflow.log_metrics(metrics)
        print({k: round(v, 2) for k, v in metrics.items()})


if __name__ == "__main__":
    main()

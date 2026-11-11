"""Load a serving bundle and turn raw sensor history into an RUL prediction."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loading import COLUMNS
from src.data.scaling import ConditionScaler
from src.data.windowing import last_window

RAW_COLUMNS = COLUMNS[2:]   # 3 operating settings + 21 sensors = 24 numbers per cycle
DEFAULT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "fd004_gru"


class RULService:
    def __init__(self, bundle_dir: Path):
        import torch
        from src.models.factory import build_model

        self.config = json.loads((bundle_dir / "config.json").read_text())
        self.scaler = ConditionScaler.from_dict(json.loads((bundle_dir / "scaler.json").read_text()))
        self.window, self.cap = self.config["window"], self.config["cap"]
        self.feature_cols = self.config["feature_cols"]
        self.model = build_model(self.config["model"], self.config["n_features"],
                                 hidden=self.config["hidden"])
        self.model.load_state_dict(torch.load(bundle_dir / "model.pt", map_location="cpu"))
        self.model.eval()
        self._torch = torch

    @classmethod
    def from_env(cls) -> "RULService":
        return cls(Path(os.environ.get("MODEL_DIR", DEFAULT_DIR)))

    def predict(self, cycles: list) -> dict:
        """cycles: oldest-to-newest list of 24-number rows (RAW_COLUMNS order)."""
        df = pd.DataFrame(cycles, columns=RAW_COLUMNS)
        scaled = self.scaler.transform(df)
        feats = scaled[self.feature_cols].to_numpy(dtype=float)
        x = last_window(feats, self.window)[None]            # (1, window, n_features)
        with self._torch.no_grad():
            out = self.model(self._torch.tensor(x, dtype=self._torch.float32)).item()
        return {"rul": float(np.clip(out * self.cap, 0, self.cap)),
                "cycles_used": len(cycles),
                "padded": len(cycles) < self.window,
                "operating_condition": int(self.scaler._assign(df.iloc[[-1]])[0])}

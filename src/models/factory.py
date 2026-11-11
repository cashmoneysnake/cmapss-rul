"""Build a neural model by name (imports torch lazily so CI without torch still works)."""


def build_model(name: str, n_features: int, hidden: int = 64):
    if name == "lstm":
        from src.models.lstm import LSTMRegressor
        return LSTMRegressor(n_features, hidden=hidden)
    if name == "gru":
        from src.models.gru import GRURegressor
        return GRURegressor(n_features, hidden=hidden)
    if name == "cnn":
        from src.models.cnn import CNNRegressor
        return CNNRegressor(n_features, hidden=hidden)
    raise ValueError(f"unknown model: {name}")

"""FastAPI service: POST sensor history, get a remaining-useful-life prediction.

Run locally:  make serve      (then open http://127.0.0.1:8000/docs)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.schemas import PredictRequest, PredictResponse
from src.api.service import RAW_COLUMNS, RULService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = RULService.from_env()   # load the model once at startup
    yield


app = FastAPI(title="C-MAPSS RUL API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.service is not None}


@app.get("/model-info")
def model_info():
    s = app.state.service
    return {"model": s.config["model"], "trained_on": s.config["subset"], "window": s.window,
            "rul_cap": s.cap, "input_columns": RAW_COLUMNS}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return app.state.service.predict(req.cycles)

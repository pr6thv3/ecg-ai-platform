from __future__ import annotations

import os
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import load_config
from src.inference import InferencePipeline


CONFIG_PATH = os.getenv("ECG_CONFIG", "configs/default.yaml")
CONFIG = load_config(CONFIG_PATH)
PIPELINE = InferencePipeline(CONFIG)
MAX_UPLOAD_BYTES = int(os.getenv("ECG_MAX_UPLOAD_BYTES", str(2 * 1024 * 1024)))
ALLOWED_EXTENSIONS = {".csv", ".txt", ".tsv"}

app = FastAPI(
    title="ECG AI System API",
    version="1.0.0",
    description=CONFIG["project"]["disclaimer"],
)

allowed_origins = [origin.strip() for origin in os.getenv("ECG_ALLOWED_ORIGINS", "").split(",") if origin.strip()]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


class PredictRequest(BaseModel):
    signal: list[float] = Field(min_length=1, max_length=200_000)
    sampling_rate: int = Field(default=360, gt=0)


class HealthResponse(BaseModel):
    status: str
    runtime: str
    model_loaded: bool
    source_mode: str
    disclaimer: str


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "runtime": PIPELINE.runtime,
        "model_loaded": PIPELINE.runtime != "demo_heuristic",
        "source_mode": CONFIG["dataset"].get("source", "unknown"),
        "disclaimer": CONFIG["project"]["disclaimer"],
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return PIPELINE.model_info()


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    try:
        return PIPELINE.predict_signal(
            np.asarray(payload.signal, dtype=np.float32),
            payload.sampling_rate,
            input_filename="json_payload",
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported ECG upload type '{suffix}'. Use CSV, TSV, or TXT.")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"ECG upload exceeds {MAX_UPLOAD_BYTES} byte limit.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=422, detail="ECG upload must be UTF-8 text.") from exc

    try:
        values = _parse_signal_text(text, suffix)
        return PIPELINE.predict_signal(
            values,
            int(CONFIG["dataset"]["sampling_rate"]),
            input_filename=file.filename or "upload",
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _parse_signal_text(text: str, suffix: str) -> np.ndarray:
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(StringIO(text), sep=delimiter)
        numeric = frame.select_dtypes(include=["number"])
        if not numeric.empty:
            return numeric.iloc[:, 0].to_numpy(dtype=np.float32)
    values = np.fromstring(text.replace(",", "\n"), sep="\n", dtype=np.float32)
    if values.size == 0:
        raise ValueError("No numeric ECG samples found in upload.")
    return values

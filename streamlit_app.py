from __future__ import annotations

import os
import tempfile
import json
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

from src.config import load_config
from src.explainability import explain_prediction
from src.inference import InferencePipeline


def _predict_via_backend(backend_url: str, input_path: str, filename: str) -> dict:
    with open(input_path, "rb") as handle:
        files = {"file": (filename, handle, "text/csv")}
        response = httpx.post(f"{backend_url}/predict-file", files=files, timeout=30.0)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="ECG AI System", page_icon="ECG", layout="wide")
config = load_config(os.getenv("ECG_CONFIG", "configs/default.yaml"))
backend_url = os.getenv("ECG_API_URL", "").rstrip("/")
pipeline = None if backend_url else InferencePipeline(config)

st.title("ECG AI System")
st.caption(config["project"]["disclaimer"])
st.warning("Research and education demo only. Do not use for clinical diagnosis or medical decisions.")

metrics_path = Path("reports/evaluation/mitbih_evaluation_summary.json")
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    status_cols = st.columns(4)
    status_cols[0].metric("Data source", metrics.get("source", "unknown"))
    status_cols[1].metric("Records", metrics.get("record_count", "n/a"))
    status_cols[2].metric("Macro F1", f"{metrics.get('macro_f1', 0):.3f}")
    status_cols[3].metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    st.caption("Displayed metrics are engineering validation artifacts, not clinical performance claims.")

uploaded = st.file_uploader("Upload ECG CSV, TSV, or TXT", type=["csv", "txt", "tsv"])
use_demo = st.button("Use demo ECG sample")

if uploaded or use_demo:
    if uploaded:
        suffix = Path(uploaded.name).suffix or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(uploaded.getvalue())
            input_path = handle.name
        display_name = uploaded.name
    else:
        input_path = config["dataset"]["sample_file"]
        display_name = "demo_ecg.csv"

    try:
        if backend_url:
            result = _predict_via_backend(backend_url, input_path, display_name)
            explanation = None
        else:
            assert pipeline is not None
            result = pipeline.predict_file(input_path)
            explanation = explain_prediction(config, input_path)

        prediction = result["prediction"]
        metric_cols = st.columns(3)
        metric_cols[0].metric("Predicted class", prediction["class_name"])
        metric_cols[1].metric("Confidence", f"{prediction['confidence']:.3f}")
        metric_cols[2].metric("Runtime", result["model"]["runtime"])

        probs = pd.DataFrame(
            [{"class": label, "probability": value} for label, value in prediction["probabilities"].items()]
        )
        st.bar_chart(probs.set_index("class"))
        st.json(result["timing_ms"])

        if explanation:
            st.image(explanation["image"], caption=f"{explanation['method']} explanation (research-only)")
        elif backend_url:
            st.info("Backend mode is active through ECG_API_URL. Local explanation rendering is disabled for remote API calls.")
    except Exception as exc:
        st.error(str(exc))
else:
    mode = f"backend: {backend_url}" if backend_url else "local Python pipeline"
    st.info(f"Upload an ECG file or use the demo sample to run inference through {mode}.")

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.config import resolve_path
from src.inference import InferencePipeline
from src.preprocessing import load_signal_file, segment_signal
from src.utils.io import write_json


def explain_prediction(config: dict[str, Any], input_path: str | Path) -> dict[str, Any]:
    pipeline = InferencePipeline(config)
    signal = load_signal_file(input_path, config, create_demo_if_missing=True)
    windows = segment_signal(signal, config)
    window = windows[0]
    base_probs = pipeline.predict_windows(window.reshape(1, -1))[0]
    pred_idx = pipeline._predict_index(base_probs)
    if pipeline.runtime == "pytorch" and getattr(pipeline, "_torch_model", None) is not None:
        saliency = _pytorch_saliency(pipeline, window, pred_idx)
        method = "gradient_saliency"
    else:
        saliency = _occlusion_saliency(pipeline, window, pred_idx)
        method = "occlusion_sensitivity"
    class_name = config["model"]["class_names"][pred_idx]

    out_dir = resolve_path(config["reports"]["explainability_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "prediction_explanation.png"
    _plot_explanation(image_path, window, saliency, class_name, float(base_probs[pred_idx]))
    summary = {
        "input": str(resolve_path(input_path)),
        "runtime": pipeline.runtime,
        "predicted_class": class_name,
        "confidence": float(base_probs[pred_idx]),
        "probabilities": {name: float(base_probs[idx]) for idx, name in enumerate(config["model"]["class_names"])},
        "thresholds_used": bool(pipeline.thresholds),
        "method": method,
        "image": str(image_path),
        "research_only": True,
        "warning": "Saliency is a technical model-debugging signal, not medical reasoning or clinical evidence.",
        "disclaimer": config["project"]["disclaimer"],
    }
    write_json(out_dir / "explanation_summary.json", summary)
    return summary


def _pytorch_saliency(pipeline: InferencePipeline, window: np.ndarray, pred_idx: int) -> np.ndarray:
    torch = getattr(pipeline, "_torch", None)
    model = getattr(pipeline, "_torch_model", None)
    if torch is None or model is None:
        return _occlusion_saliency(pipeline, window, pred_idx)
    tensor = torch.tensor(window.reshape(1, -1), dtype=torch.float32, requires_grad=True)
    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    logits[0, pred_idx].backward()
    grad = tensor.grad.detach().cpu().numpy().reshape(-1)
    saliency = np.abs(grad).astype(np.float32)
    max_val = float(np.max(saliency))
    return saliency / max_val if max_val > 0 else saliency


def _occlusion_saliency(pipeline: InferencePipeline, window: np.ndarray, pred_idx: int) -> np.ndarray:
    baseline = float(pipeline.predict_windows(window.reshape(1, -1))[0, pred_idx])
    saliency = np.zeros_like(window, dtype=np.float32)
    segment = max(8, len(window) // 30)
    for start in range(0, len(window), segment):
        occluded = window.copy()
        occluded[start : start + segment] = 0.0
        score = float(pipeline.predict_windows(occluded.reshape(1, -1))[0, pred_idx])
        saliency[start : start + segment] = max(0.0, baseline - score)
    max_val = float(np.max(saliency))
    return saliency / max_val if max_val > 0 else saliency


def _plot_explanation(path: Path, window: np.ndarray, saliency: np.ndarray, class_name: str, confidence: float) -> None:
    x = np.arange(len(window))
    plt.figure(figsize=(10, 4))
    plt.plot(x, window, color="#0f172a", linewidth=1.2, label="ECG window")
    plt.fill_between(x, np.min(window), np.max(window), color="#ef4444", alpha=saliency * 0.35, label="Occlusion saliency")
    plt.title(f"Prediction explanation: {class_name} ({confidence:.2f})")
    plt.xlabel("Sample")
    plt.ylabel("Normalized amplitude")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

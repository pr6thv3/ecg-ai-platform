from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import resolve_path
from src.monitoring import InferenceLogger
from src.preprocessing import load_signal_file, segment_signal, validate_signal


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


class InferencePipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.class_names = list(config["model"]["class_names"])
        self.input_size = int(config["model"]["input_size"])
        self.runtime = "unloaded"
        self.model_hash: str | None = None
        self.requested_checkpoint = str(resolve_path(config["model"]["checkpoint"]))
        self.loaded_checkpoint: str | None = None
        self.fallback_used = False
        self.model_type = "none"
        self.model_warnings: list[str] = []
        self.thresholds = self._load_thresholds()
        self._onnx_session = None
        self._torch_model = None
        self._torch = None
        self._load_model()
        self.logger = InferenceLogger(config["artifacts"]["inference_log"])

    def _load_model(self) -> None:
        checkpoint = resolve_path(self.config["model"]["checkpoint"])
        fallback = resolve_path(self.config["model"].get("fallback_checkpoint", "artifacts/models/best_model.pt"))
        runtime = str(self.config["model"].get("runtime", "auto")).lower()
        candidates: list[tuple[Path, bool]] = []
        if checkpoint.exists():
            candidates.append((checkpoint, False))
        else:
            self._warn(f"Configured checkpoint is missing: {checkpoint}")
        if self.config["model"].get("allow_fallback_checkpoint", False) and fallback.exists() and fallback != checkpoint:
            candidates.append((fallback, True))

        for candidate, fallback_used in candidates:
            suffix = candidate.suffix.lower()
            if suffix == ".onnx":
                if runtime == "pytorch" and not fallback_used:
                    self._warn(f"Checkpoint is ONNX but model.runtime=pytorch; loading ONNX from {candidate}.")
                if runtime == "pytorch" and fallback_used:
                    self._warn(f"Using ONNX fallback because PyTorch checkpoint was unavailable: {candidate}.")
                if self._load_onnx(candidate):
                    self.fallback_used = fallback_used
                    return
            if suffix in {".pt", ".pth"}:
                if runtime == "onnx" and not fallback_used:
                    self._warn(f"Checkpoint is PyTorch but model.runtime=onnx; loading PyTorch from {candidate}.")
                if self._load_pytorch(candidate):
                    self.fallback_used = fallback_used
                    return

        if not self.config["model"].get("allow_missing_checkpoint", False):
            raise FileNotFoundError(
                f"No supported model checkpoint loaded. Expected {checkpoint}. "
                "Set ECG_ALLOW_MISSING_CHECKPOINT=true only for CI/demo smoke tests."
            )
        self.runtime = "demo_heuristic"
        self.model_hash = None
        self.loaded_checkpoint = None
        self.model_type = "demo_heuristic"
        self._warn("No supported model checkpoint loaded; using deterministic demo heuristic because it is explicitly enabled.")

    def _load_onnx(self, checkpoint: Path) -> bool:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            self._warn(f"onnxruntime is unavailable, cannot load {checkpoint}: {exc}")
            return False

        providers = [self.config["model"].get("onnx_provider", "CPUExecutionProvider")]
        self._onnx_session = ort.InferenceSession(str(checkpoint), providers=providers)
        self.runtime = "onnx"
        self.model_type = "onnx"
        self.loaded_checkpoint = str(checkpoint)
        self.model_hash = _sha256(checkpoint)
        return True

    def _load_pytorch(self, checkpoint: Path) -> bool:
        try:
            import torch

            from src.models import build_model
        except ImportError as exc:
            self._warn(f"PyTorch is unavailable, cannot load {checkpoint}: {exc}")
            return False

        self._torch = torch
        payload = torch.load(str(checkpoint), map_location="cpu")
        if isinstance(payload, dict) and "model_state_dict" in payload:
            self._validate_checkpoint_metadata(payload, checkpoint)
            state_dict = payload["model_state_dict"]
            model_type = _normalize_model_type(str(payload.get("model_type") or self.config["model"].get("type", "baseline_cnn")))
        else:
            self._warn(f"Legacy PyTorch checkpoint has no metadata guard fields: {checkpoint}")
            state_dict = payload
            model_type = _normalize_model_type(str(self.config["model"].get("type", "baseline_cnn")))

        model = build_model(
            model_type,
            num_classes=len(self.class_names),
            input_size=self.input_size,
            dropout=float(self.config["model"].get("dropout", 0.25)),
        )
        model.load_state_dict(state_dict)
        model.eval()
        self._torch_model = model
        self.runtime = "pytorch"
        self.model_type = model_type
        self.loaded_checkpoint = str(checkpoint)
        self.model_hash = _sha256(checkpoint)
        return True

    def _validate_checkpoint_metadata(self, payload: dict[str, Any], checkpoint: Path) -> None:
        checks: list[str] = []
        expected_type = str(self.config["model"].get("type", "baseline_cnn"))
        payload_type = payload.get("model_type")
        normalized_payload_type = _normalize_model_type(str(payload_type)) if payload_type else None
        if normalized_payload_type and normalized_payload_type != expected_type:
            checks.append(f"model_type={payload_type} expected {expected_type}")
        checkpoint_classes = payload.get("class_order") or payload.get("class_names")
        if checkpoint_classes and list(checkpoint_classes) != self.class_names:
            checks.append(f"class_order={checkpoint_classes} expected {self.class_names}")
        if payload.get("input_size") and int(payload["input_size"]) != self.input_size:
            checks.append(f"input_size={payload.get('input_size')} expected {self.input_size}")
        expected_mapping = self.config["dataset"].get("class_mapping", {})
        if payload.get("class_mapping") and dict(payload["class_mapping"]) != dict(expected_mapping):
            checks.append("class_mapping differs from config dataset.class_mapping")
        preprocessing = payload.get("preprocessing", {})
        if preprocessing:
            if int(preprocessing.get("window_size", self.input_size)) != int(self.config["preprocessing"]["window_size"]):
                checks.append("preprocessing.window_size differs from config")
            if int(preprocessing.get("sampling_rate", self.config["dataset"]["sampling_rate"])) != int(self.config["dataset"]["sampling_rate"]):
                checks.append("preprocessing.sampling_rate differs from config")
            expected_norm = str(self.config["preprocessing"].get("normalization", "maxabs"))
            checkpoint_norm = str(preprocessing.get("normalization", expected_norm))
            if checkpoint_norm != expected_norm:
                checks.append("preprocessing.normalization differs from config")

        if not checks:
            return
        message = f"Checkpoint metadata mismatch for {checkpoint}: {'; '.join(checks)}"
        if self.config["model"].get("checkpoint_validation", True):
            raise ValueError(message)
        self._warn(message)

    def _warn(self, message: str) -> None:
        self.model_warnings.append(message)
        print(f"WARNING: {message}")

    def model_info(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "model_type": self.model_type,
            "model_hash": self.model_hash,
            "requested_checkpoint": self.requested_checkpoint,
            "loaded_checkpoint": self.loaded_checkpoint,
            "fallback_used": self.fallback_used,
            "thresholds_loaded": bool(self.thresholds),
            "warnings": self.model_warnings,
            "class_names": self.class_names,
            "input_size": self.input_size,
            "disclaimer": self.config["project"]["disclaimer"],
        }

    def predict_file(self, input_path: str | Path) -> dict[str, Any]:
        signal = load_signal_file(input_path, self.config, create_demo_if_missing=True)
        return self.predict_signal(
            signal,
            sampling_rate=int(self.config["dataset"]["sampling_rate"]),
            input_filename=str(input_path),
        )

    def predict_signal(self, signal: np.ndarray, sampling_rate: int, input_filename: str | None = None) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            validated = validate_signal(signal, sampling_rate, self.config)
            pre_start = time.perf_counter()
            windows = segment_signal(validated, self.config)
            pre_ms = (time.perf_counter() - pre_start) * 1000
            inf_start = time.perf_counter()
            probabilities = self.predict_windows(windows)
            inf_ms = (time.perf_counter() - inf_start) * 1000
            predictions = self._format_predictions(probabilities)
            elapsed = (time.perf_counter() - start) * 1000
            top = predictions[0]
            result = {
                "status": "ok",
                "model": self.model_info(),
                "input": {
                    "filename": input_filename or "",
                    "samples": int(validated.size),
                    "sampling_rate": sampling_rate,
                    "window_count": int(windows.shape[0]),
                    "source_mode": self._source_mode(),
                },
                "predicted_class": top["class_name"],
                "confidence": top["confidence"],
                "probabilities": top["probabilities"],
                "thresholds_used": bool(self.thresholds),
                "warning": config_warning(self.config),
                "disclaimer": self.config["project"]["disclaimer"],
                "prediction": top,
                "predictions": predictions,
                "timing_ms": {
                    "preprocessing": round(pre_ms, 4),
                    "inference": round(inf_ms, 4),
                    "total": round(elapsed, 4),
                },
            }
            self._log_result(result, elapsed, "ok", "")
            return result
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            self._log_result(
                {
                    "model": self.model_info(),
                    "input": {
                        "filename": input_filename or "",
                        "samples": int(np.asarray(signal).size),
                        "window_count": 0,
                        "source_mode": self._source_mode(),
                    },
                    "prediction": {"class_name": "", "confidence": 0.0},
                },
                elapsed,
                "error",
                str(exc),
            )
            raise

    def predict_windows(self, windows: np.ndarray) -> np.ndarray:
        arr = np.asarray(windows, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != self.input_size:
            raise ValueError(f"Expected ECG windows with shape (n, {self.input_size}), got {arr.shape}.")
        if not np.isfinite(arr).all():
            raise ValueError("ECG windows contain NaN/Inf values.")

        if self.runtime == "onnx" and self._onnx_session is not None:
            input_name = self._onnx_session.get_inputs()[0].name
            logits = self._onnx_session.run(None, {input_name: arr})[0]
            return _softmax(logits)

        if self.runtime == "pytorch" and self._torch_model is not None and self._torch is not None:
            batch_size = max(1, int(self.config.get("inference", {}).get("batch_size", 2048)))
            outputs = []
            with self._torch.no_grad():
                for start in range(0, arr.shape[0], batch_size):
                    batch = self._torch.tensor(arr[start : start + batch_size], dtype=self._torch.float32)
                    outputs.append(self._torch_model(batch).detach().cpu().numpy())
            return _softmax(np.concatenate(outputs, axis=0))

        return self._heuristic_probabilities(arr)

    def _heuristic_probabilities(self, windows: np.ndarray) -> np.ndarray:
        probs = []
        for window in windows:
            width = float(np.mean(np.abs(np.diff(window))) / (np.std(window) + 1e-6))
            peak = float(np.max(window) - np.min(window))
            scores = np.array(
                [
                    1.0,
                    0.6 + peak,
                    0.5 + abs(float(np.argmax(window)) / len(window) - 0.45),
                    0.4 + width,
                    0.4 + max(0.0, 1.0 - width),
                ],
                dtype=np.float32,
            )
            probs.append(_softmax(scores.reshape(1, -1))[0])
        return np.asarray(probs, dtype=np.float32)

    def _format_predictions(self, probabilities: np.ndarray) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for idx, row in enumerate(probabilities):
            pred_idx = self._predict_index(row)
            confidence = float(row[pred_idx])
            formatted.append(
                {
                    "window_index": idx,
                    "class_index": pred_idx,
                    "class_name": self.class_names[pred_idx],
                    "confidence": confidence,
                    "low_confidence": confidence < float(self.config["inference"]["low_confidence_threshold"]),
                    "thresholds_used": bool(self.thresholds),
                    "probabilities": {name: float(row[i]) for i, name in enumerate(self.class_names)},
                }
            )
        formatted.sort(key=lambda item: item["confidence"], reverse=True)
        return formatted

    def _predict_index(self, row: np.ndarray) -> int:
        if not self.thresholds:
            return int(np.argmax(row))
        threshold_values = np.asarray([float(self.thresholds.get(name, 1.0)) for name in self.class_names], dtype=np.float32)
        margins = row - threshold_values
        eligible = np.where(margins >= 0)[0]
        if eligible.size:
            return int(eligible[np.argmax(margins[eligible])])
        return int(np.argmax(row))

    def _load_thresholds(self) -> dict[str, float]:
        threshold_path = self.config["model"].get("threshold_path")
        if not threshold_path:
            return {}
        path = resolve_path(threshold_path)
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            thresholds = payload.get("thresholds", payload)
            return {str(key): float(value) for key, value in thresholds.items() if str(key) in self.class_names}
        except Exception as exc:
            self._warn(f"Failed to load thresholds from {path}: {exc}")
            return {}

    def _source_mode(self) -> str:
        if self.runtime == "demo_heuristic":
            return "synthetic_or_demo_heuristic"
        return str(self.config["dataset"].get("source", "unknown"))

    def _log_result(self, result: dict[str, Any], elapsed_ms: float, status: str, error: str) -> None:
        if not self.config["inference"].get("log_inference", True):
            return
        prediction = result.get("prediction", {})
        input_meta = result.get("input", {})
        self.logger.log(
            {
                "runtime": result.get("model", {}).get("runtime", self.runtime),
                "input_filename": input_meta.get("filename", ""),
                "input_samples": input_meta.get("samples", 0),
                "input_length": input_meta.get("samples", 0),
                "window_count": input_meta.get("window_count", 0),
                "prediction": prediction.get("class_name", ""),
                "confidence": prediction.get("confidence", 0.0),
                "processing_time_ms": round(elapsed_ms, 4),
                "latency_ms": round(elapsed_ms, 4),
                "status": status,
                "error_flag": bool(error),
                "error": error,
                "source_mode": input_meta.get("source_mode", self._source_mode()),
            }
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_warning(config: dict[str, Any]) -> str:
    return config["project"]["disclaimer"]


def _normalize_model_type(model_type: str) -> str:
    return {"ECGNet": "baseline_cnn", "BaselineCNN": "baseline_cnn"}.get(model_type, model_type)

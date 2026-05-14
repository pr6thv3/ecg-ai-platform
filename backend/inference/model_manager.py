import time
import threading
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

from config.settings import settings
from models.train import ECGNet
try:
    from explainability.grad_cam import GradCAM1D
except ImportError:
    GradCAM1D = None

from utils.logger import get_logger
from monitoring.metrics import INFERENCE_LATENCY, INFERENCE_ERRORS

logger = get_logger("inference.model_manager")

AAMI_CLASSES = ['N', 'V', 'A', 'L', 'R']
AAMI_LABELS = {
    0: "Normal (N)",
    1: "PVC (V)",
    2: "APB (A)",
    3: "LBBB (L)",
    4: "RBBB (R)"
}

USE_ONNX = settings.USE_ONNX

class ModelManager:
    """
    Thread-safe singleton managing ECG model inference.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance.model = None
                cls._instance.device = None
                cls._instance.grad_cam = None
                cls._instance.onnx_session = None
                cls._instance.model_path = None
                cls._instance.model_hash = None
                cls._instance.model_runtime = "unloaded"
                cls._instance.inference_lock = threading.Lock()
                cls._instance.warmup_latency_ms = None
        return cls._instance

    def is_loaded(self) -> bool:
        """Checks if the model has been successfully loaded."""
        return self.onnx_session is not None or self.model is not None

    def load_model(self, path: str, device: torch.device) -> None:
        """
        Loads either an ONNX artifact or a PyTorch checkpoint.

        ONNX is the production path. A sibling .pth file is loaded only when it
        exists so Grad-CAM can run; missing PyTorch weights do not block ONNX
        inference.
        """
        with self._lock:
            model_path = Path(path)
            self.model = None
            self.grad_cam = None
            self.onnx_session = None
            self.model_runtime = "unloaded"
            self.model_path = str(model_path)
            self.model_hash = None

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at: {model_path}")

            self.model_hash = self._sha256(model_path)
            if settings.MODEL_SHA256 and self.model_hash.lower() != settings.MODEL_SHA256.lower():
                raise RuntimeError(
                    f"Model hash mismatch for {model_path}. Expected {settings.MODEL_SHA256}, got {self.model_hash}"
                )

            self.device = device

            if model_path.suffix.lower() == ".onnx":
                self._load_onnx(model_path)
                pth_path = model_path.with_suffix(".pth")
                if pth_path.exists():
                    self._load_pytorch(pth_path, device)
                else:
                    logger.info("No sibling PyTorch checkpoint found; Grad-CAM is disabled for this runtime.")
                self.model_runtime = "onnx"
                return

            self._load_pytorch(model_path, device)
            onnx_path = model_path.with_suffix(".onnx")
            if onnx_path.exists():
                self._load_onnx(onnx_path)
                self.model_runtime = "onnx" if USE_ONNX else "pytorch"
            else:
                logger.warning(f"ONNX model file not found at {onnx_path}.")
                self.model_runtime = "pytorch"

    def _load_pytorch(self, path: Path, device: torch.device) -> None:
        logger.info(f"Loading ECGNet from {path} onto {device}...")
        self.model = ECGNet(num_classes=5)
        self.model.load_state_dict(torch.load(str(path), map_location=device, weights_only=True))
        self.model.to(device)
        self.model.eval()
        self.device = device

        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"PyTorch model loaded successfully. Total trainable parameters: {param_count:,}")

        if GradCAM1D is not None:
            self.grad_cam = GradCAM1D(self.model)
            logger.info("GradCAM module initialized successfully.")
        else:
            logger.warning("GradCAM1D not available. Explainability will be disabled.")

    def _load_onnx(self, path: Path) -> None:
        logger.info(f"Loading ONNX session from {path}...")
        try:
            import onnxruntime as ort
            self.onnx_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            logger.info("ONNX model loaded successfully.")
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required to load .onnx models.") from exc

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def warmup(self) -> None:
        """
        Runs a dummy inference pass to compile CUDA kernels or pre-allocate memory buffers.
        """
        if not self.is_loaded():
            logger.warning("Attempted to warmup uninitialized model.")
            return

        with self.inference_lock:
            logger.info("Starting model warmup...")
            start = time.perf_counter()
            if USE_ONNX and hasattr(self, "onnx_session") and self.onnx_session:
                dummy_input = np.zeros((1, 360), dtype=np.float32)
                input_name = self.onnx_session.get_inputs()[0].name
                _ = self.onnx_session.run(None, {input_name: dummy_input})
            else:
                if self.model is None:
                    logger.warning("No PyTorch model is available for warmup fallback.")
                    return
                dummy_input = torch.zeros(1, 360, device=self.device)
                with torch.no_grad():
                    _ = self.model(dummy_input)
            latency = (time.perf_counter() - start) * 1000
            self.warmup_latency_ms = latency
            logger.info(f"Model warmup complete. First pass latency: {latency:.2f} ms")

    def predict(self, beat_window: np.ndarray) -> Dict[str, Any]:
        """
        Runs a forward pass on a single beat window.
        Args:
            beat_window: np.ndarray of shape (360,)
        Returns:
            Dict containing predicted class, confidence, raw probabilities, and latency.
        """
        if self.model is None:
            raise RuntimeError("PyTorch model is not loaded.")

        if beat_window.shape != (360,):
            raise ValueError(f"Invalid input shape. Expected (360,), got {beat_window.shape}")

        if np.isnan(beat_window).any() or np.isinf(beat_window).any():
            raise ValueError("Input contains NaN or infinite values.")

        try:
            start = time.perf_counter()
            with self.inference_lock:
                # PyTorch inference
                tensor_input = torch.tensor(beat_window, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(tensor_input)
                    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                        
            latency = (time.perf_counter() - start) * 1000
            INFERENCE_LATENCY.labels(runtime="pytorch").observe(latency / 1000.0)
            if latency > 50:
                logger.warning("Slow inference detected", extra={"latency_ms": latency})

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            
            probabilities = {AAMI_CLASSES[i]: float(probs[i]) for i in range(len(AAMI_CLASSES))}

            return {
                "beat_type": AAMI_CLASSES[pred_idx],
                "confidence": confidence,
                "probabilities": probabilities,
                "latency_ms": latency
            }
        except Exception as e:
            INFERENCE_ERRORS.inc()
            logger.error(f"Inference failed: {e}")
            raise e

    def classify(self, beat_window: np.ndarray) -> Dict[str, Any]:
        """Runs inference with the configured runtime and falls back to PyTorch when ONNX is unavailable."""
        if USE_ONNX and self.onnx_session is not None:
            return self.onnx_predict(beat_window)
        return self.predict(beat_window)

    def onnx_predict(self, beat_window: np.ndarray) -> Dict[str, Any]:
        """
        Runs a forward pass on a single beat window using ONNX Runtime.
        """
        if not hasattr(self, "onnx_session") or self.onnx_session is None:
            raise RuntimeError("ONNX model is not loaded.")

        if beat_window.shape != (360,):
            raise ValueError(f"Invalid input shape. Expected (360,), got {beat_window.shape}")
        if np.isnan(beat_window).any() or np.isinf(beat_window).any():
            raise ValueError("Input contains NaN or infinite values.")

        try:
            start = time.perf_counter()
            with self.inference_lock:
                # Prepare ONNX input
                input_name = self.onnx_session.get_inputs()[0].name
                onnx_input = np.array(beat_window, dtype=np.float32).reshape(1, 360)
                
                # ONNX Inference
                logits = self.onnx_session.run(None, {input_name: onnx_input})[0]
                exp_logits = np.exp(logits[0] - np.max(logits[0]))
                probs = exp_logits / exp_logits.sum()

            latency = (time.perf_counter() - start) * 1000
            INFERENCE_LATENCY.labels(runtime="onnx").observe(latency / 1000.0)
            if latency > 50:
                logger.warning("Slow ONNX inference detected", extra={"latency_ms": latency})

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            probabilities = {AAMI_CLASSES[i]: float(probs[i]) for i in range(len(AAMI_CLASSES))}

            return {
                "beat_type": AAMI_CLASSES[pred_idx],
                "confidence": confidence,
                "probabilities": probabilities,
                "latency_ms": latency
            }
        except Exception as e:
            INFERENCE_ERRORS.inc()
            logger.error(f"ONNX Inference failed: {e}")
            raise e

    def _get_dominant_region(self, peak_idx: int) -> str:
        """Heuristic to map saliency peak to physiological waveform region."""
        if peak_idx < 150:
            return "P-wave"
        elif peak_idx <= 210:
            return "QRS Complex"
        elif peak_idx <= 250:
            return "ST-segment"
        else:
            return "T-wave"

    def explain(self, beat_window: np.ndarray, target_class: int) -> Dict[str, Any]:
        """
        Generates a Grad-CAM saliency map for the provided beat window.
        Args:
            beat_window: np.ndarray of shape (360,)
            target_class: Integer index of the class to explain.
        Returns:
            Dict containing the saliency map list and the dominant physiological region.
        """
        if not self.is_loaded() or self.grad_cam is None:
            raise RuntimeError("Model or GradCAM is not loaded.")

        try:
            tensor_input = torch.tensor(beat_window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            tensor_input = tensor_input.to(self.device)

            with self.inference_lock:
                # generate() likely returns a normalized numpy array
                saliency_map, top_preds = self.grad_cam.generate(tensor_input, target_class)
            
            # Map saliency arrays to lists for JSON serialization
            if isinstance(saliency_map, np.ndarray):
                saliency_list = saliency_map.tolist()
            else:
                saliency_list = list(saliency_map)
                
            peak_idx = int(np.argmax(saliency_list))
            dominant_region = self._get_dominant_region(peak_idx)

            return {
                "saliency_map": saliency_list,
                "dominant_region": dominant_region,
                "predictions": top_preds # Assumes grad_cam returns the prediction dictionary format
            }
            
        except Exception as e:
            logger.error(f"Explainability failed: {e}")
            raise e

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
import asyncio
import numpy as np
import time
import os
import sys

# Ensure backend root is in path for relative imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.connection_manager import ConnectionManager
from config.settings import settings
from utils.stream_state import StreamStateTracker
from utils.ecg_simulator import ECGSimulator
from inference.model_manager import ModelManager
from pydantic import BaseModel, Field, field_validator
from typing import List

from utils.logger import get_logger
from monitoring.metrics import WS_CONNECTIONS, WS_ERRORS, BEATS_CLASSIFIED, WS_MESSAGES_RECEIVED, dashboard_tracker, PIPELINE_STAGE_LATENCY

logger = get_logger("api.ws_router")

router = APIRouter()
manager = ConnectionManager()

class ExplainRequest(BaseModel):
    beat_window: List[float] = Field(
        min_length=settings.MAX_BEAT_WINDOW_LENGTH,
        max_length=settings.MAX_BEAT_WINDOW_LENGTH,
    )
    predicted_class: int = Field(ge=0, le=4)

    @field_validator("beat_window")
    @classmethod
    def validate_beat_window(cls, value: List[float]) -> List[float]:
        if any(not np.isfinite(sample) for sample in value):
            raise ValueError("beat_window must contain only finite numbers")
        return value

class AnalyzeRequest(BaseModel):
    beat_window: List[float] = Field(
        min_length=settings.MAX_BEAT_WINDOW_LENGTH,
        max_length=settings.MAX_BEAT_WINDOW_LENGTH,
    )

    @field_validator("beat_window")
    @classmethod
    def validate_beat_window(cls, value: List[float]) -> List[float]:
        if any(not np.isfinite(sample) for sample in value):
            raise ValueError("beat_window must contain only finite numbers")
        return value

def get_dominant_region(peak_idx: int) -> str:
    if peak_idx < 150:
        return "P-wave"
    elif peak_idx <= 210:
        return "QRS Complex"
    elif peak_idx <= 250:
        return "ST-segment"
    else:
        return "T-wave"


@router.websocket("/ws/ecg-stream")
async def websocket_ecg_endpoint(websocket: WebSocket, mode: str = "synthetic", pattern: str = "normal", record: str = "100"):
    """
    Main WebSocket endpoint managing the real-time inference lifecycle.
    Args:
        mode: 'synthetic' or 'mitbih'
        pattern: Event injection pattern if synthetic ('normal', 'pvc_burst', 'apb')
        record: MIT-BIH record number if mitbih mode
    """
    origin = websocket.headers.get("origin")
    if origin and origin not in settings.allowed_origins_list:
        await websocket.close(code=1008)
        return

    if mode not in {"synthetic", "mitbih"}:
        await websocket.close(code=1008)
        return
    if pattern not in {"normal", "pvc_burst", "apb"}:
        await websocket.close(code=1008)
        return
    if mode == "mitbih" and record not in settings.allowed_mitbih_records_set:
        await websocket.close(code=1008)
        return

    await manager.connect(websocket)
    WS_CONNECTIONS.inc()
    logger.info("WebSocket connection opened", extra={"mode": mode, "pattern": pattern, "record": record})
    tracker = StreamStateTracker()
    simulator = ECGSimulator()
    
    last_timestamp = time.time()
    
    try:
        # 1. Initialize the correct data stream
        if mode == "mitbih":
            stream = simulator.stream_mitbih(f"../datasets/mit-bih/{record}")
        else:
            stream = simulator.stream_synthetic(pattern=pattern)
            
        async for beat_data in stream:
            # 2. Preprocessing
            preprocess_started = time.perf_counter()
            raw_window = np.array(beat_data["raw_window"], dtype=np.float32)
            PIPELINE_STAGE_LATENCY.labels(stage="payload_preprocess").observe(time.perf_counter() - preprocess_started)
            
            # 3. Model Inference (Real)
            try:
                model_mgr = ModelManager()
                result = model_mgr.classify(raw_window)
                beat_type = result["beat_type"]
                confidence = result["confidence"]
                error_state = None
                
                BEATS_CLASSIFIED.labels(beat_type=beat_type).inc()
                
                logger.info("Beat classified", extra={"beat_type": beat_type, "confidence": confidence})
                
                if confidence < 0.6:
                    logger.warning("Low confidence inference", extra={"beat_type": beat_type, "confidence": confidence})
                
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                beat_type = "UNKNOWN"
                confidence = 0.0
                error_state = "inference_failed"
            
            # 4. State Tracking & BPM
            current_time = beat_data["timestamp"]
            rr_interval = current_time - last_timestamp
            last_timestamp = current_time
            
            tracker.add_beat(rr_interval_sec=rr_interval, confidence=confidence, is_pvc=(beat_type == "V"))
            alerts = tracker.check_alerts(confidence)
            
            # Record dashboard metrics
            latency_ms = result.get("latency_ms", 0.0) if not error_state else 0.0
            dashboard_tracker.record_inference(latency_ms=latency_ms, is_alert=bool(alerts))
            
            # 5. Build and Emit JSON Payload
            payload = {
                "timestamp": current_time,
                "bpm": round(tracker.current_bpm, 1),
                "beat_type": beat_type,
                "confidence": round(confidence, 3),
                "rhythm_class": tracker.rhythm_classification,
                "anomaly_score": round(tracker.anomaly_score, 3),
                "raw_window": beat_data["raw_window"], # Sent for frontend rendering
                "alert": alerts if alerts else None
            }
            
            if error_state:
                payload["error"] = error_state
            
            send_started = time.perf_counter()
            await manager.send_personal_message(payload, websocket)
            PIPELINE_STAGE_LATENCY.labels(stage="websocket_send").observe(time.perf_counter() - send_started)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        WS_CONNECTIONS.dec()
        logger.info("Client gracefully disconnected from ECG stream.")
    except asyncio.CancelledError:
        manager.disconnect(websocket)
        WS_CONNECTIONS.dec()
        logger.info("WebSocket connection cancelled.")
    except Exception as e:
        manager.disconnect(websocket)
        WS_CONNECTIONS.dec()
        WS_ERRORS.inc()
        logger.error(f"WebSocket Error: {e}")

@router.post("/explain")
async def explain_beat(request: ExplainRequest):
    """
    Grad-CAM explainability endpoint.
    Takes a raw beat window and target class, returning the saliency map.
    """
    raw_window = np.array(request.beat_window, dtype=np.float32)
    
    try:
        model_mgr = ModelManager()
        result = model_mgr.explain(raw_window, request.predicted_class)
        return result
    except Exception as e:
        logger.error(f"Explainability failed: {e}")
        raise HTTPException(status_code=503, detail="Explainability is unavailable for the loaded model.") from e

@router.post("/analyze")
async def analyze_beat(request: AnalyzeRequest):
    """
    Synchronous single-beat inference endpoint.
    Accepts a 360-sample beat window, returns classification result.
    """
    raw_window = np.array(request.beat_window, dtype=np.float32)
    
    try:
        model_mgr = ModelManager()
        return model_mgr.classify(raw_window)
    except Exception as e:
        logger.error(f"Analyze failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

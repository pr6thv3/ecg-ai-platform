from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time

from api.ws_router import router as ws_router
from api.report_router import router as report_router

from config.settings import settings
from utils.logger import setup_logging, get_logger, RequestIDMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

from monitoring.metrics import dashboard_tracker, WS_CONNECTIONS

setup_logging()
logger = get_logger("api.main")

from inference.model_manager import ModelManager
from inference.device_manager import get_device

START_TIME = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Loads the model once on startup and warms it up to prevent latency spikes on first request.
    """
    logger.info("Application startup initiated.")
    
    # Load the model
    model_path = str(settings.MODEL_PATH)
    
    try:
        device = get_device()
        manager = ModelManager()
        manager.load_model(model_path, device)
        manager.warmup()
    except FileNotFoundError as e:
        logger.error(f"Model unavailable at startup: {e}")
    except Exception as e:
        logger.error(f"Model loading failed at startup: {e}")
        
    yield
    
    # Cleanup on shutdown
    logger.info("Application shutdown initiated.")

# Initialize FastAPI with lifespan
app = FastAPI(
    title="ECG AI Platform",
    description="Real-time ECG Arrhythmia Classification System",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
# Allow Vercel deployment URLs and localhost for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"], # Allow all headers
)

# Request ID middleware
app.add_middleware(RequestIDMiddleware)

# Prometheus metrics security
security = HTTPBasic(auto_error=False)

def verify_metrics_auth(credentials: HTTPBasicCredentials | None = Depends(security)):
    if not settings.METRICS_AUTH_TOKEN:
        return True

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Metrics authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    is_user_ok = secrets.compare_digest(credentials.username, "admin")
    is_pass_ok = secrets.compare_digest(credentials.password, settings.METRICS_AUTH_TOKEN)
    
    if not (is_user_ok and is_pass_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True

@app.get("/metrics", include_in_schema=False)
def get_metrics(authenticated: bool = Depends(verify_metrics_auth)):
    """
    Prometheus metrics endpoint. Secured via Basic Auth if METRICS_AUTH_TOKEN is set.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/metrics/dashboard", tags=["System"])
def get_dashboard_metrics(authenticated: bool = Depends(verify_metrics_auth)):
    """
    Returns a JSON summary of the last 60 seconds of metrics:
    inference count, average latency, alert count, and active connections.
    Suitable for frontend dashboard display.
    """
    active_conns = int(WS_CONNECTIONS._value.get())
    return dashboard_tracker.get_summary(active_conns)

# Include routers
app.include_router(ws_router)
app.include_router(report_router)

@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint for Render/Railway/Docker.
    Returns deployment status signal including model details.
    """
    manager = ModelManager()
    
    device_str = getattr(manager.device, "type", str(manager.device or "cpu"))
        
    return {
        "status": "ok" if manager.is_loaded() else "model_unavailable",
        "model_loaded": manager.is_loaded(),
        "model_type": manager.model_runtime,
        "model_path": manager.model_path,
        "model_sha256": manager.model_hash,
        "device": device_str,
        "warmup_latency_ms": manager.warmup_latency_ms,
        "version": app.version,
        "uptime_seconds": time.time() - START_TIME
    }

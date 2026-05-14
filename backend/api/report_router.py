import base64

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Optional
import uuid

from config.settings import settings
from reports.report_generator import ReportGenerator

router = APIRouter(prefix="/report", tags=["Clinical Reports"])

# In-memory store for session JSONs for the GET endpoint
# In a real app, this would be a Postgres or MongoDB database
SESSION_DB = {}
bearer_auth = HTTPBearer(auto_error=False)

def verify_report_access(credentials: HTTPAuthorizationCredentials | None = Depends(bearer_auth)):
    if not settings.REPORT_AUTH_TOKEN:
        return True
    if credentials is None or credentials.scheme.lower() != "bearer" or credentials.credentials != settings.REPORT_AUTH_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Report access token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

# --- Pydantic Models for Validation ---

class PatientMetadata(BaseModel):
    id: str = Field(min_length=1, max_length=80)
    age: int = Field(ge=0, le=130)
    gender: str = Field(min_length=1, max_length=40)
    session_date: str = Field(min_length=1, max_length=40)

class SignalMetadata(BaseModel):
    duration_sec: float = Field(gt=0, le=172800)
    sampling_rate: int = Field(gt=0, le=2000)
    snr_before: float
    snr_after: float

class ClassDistribution(BaseModel):
    N: int = Field(ge=0)
    V: int = Field(ge=0)
    A: int = Field(ge=0)
    L: int = Field(ge=0)
    R: int = Field(ge=0)

class BeatStatistics(BaseModel):
    total_beats: int = Field(ge=0)
    class_distribution: ClassDistribution
    dominant_rhythm: str = Field(min_length=1, max_length=80)

class AnomalyEvent(BaseModel):
    timestamp: float
    beat_type: str = Field(min_length=1, max_length=20)
    confidence: float = Field(ge=0, le=1)
    alert_message: str = Field(min_length=1, max_length=240)

class ModelMetrics(BaseModel):
    average_confidence: float = Field(ge=0, le=1)
    low_confidence_beats: int = Field(ge=0)

class SessionData(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    session_id: Optional[str] = Field(default=None, max_length=120)
    patient_metadata: PatientMetadata
    signal_metadata: SignalMetadata
    beat_statistics: BeatStatistics
    anomaly_events: List[AnomalyEvent] = Field(max_length=settings.MAX_REPORT_EVENTS)
    model_metrics: ModelMetrics
    waveform_b64: Optional[str] = None
    confusion_matrix_b64: Optional[str] = None

    @field_validator("waveform_b64", "confusion_matrix_b64")
    @classmethod
    def validate_image_size(cls, value: Optional[str]) -> Optional[str]:
        if not value:
            return value
        payload = value.split(",", 1)[1] if "," in value else value
        try:
            decoded_len = len(base64.b64decode(payload, validate=True))
        except Exception as exc:
            raise ValueError("image fields must be valid base64") from exc
        if decoded_len > settings.MAX_REPORT_IMAGE_BYTES:
            raise ValueError("image fields exceed the configured byte limit")
        return value


@router.post("/generate")
async def generate_report(data: SessionData, authenticated: bool = Depends(verify_report_access)):
    """
    Generates a professional clinical PDF report from ECG session data.
    """
    try:
        # Assign an ID if none exists and store for the GET endpoint
        session_id = data.session_id or str(uuid.uuid4())
        session_dict = data.model_dump()
        session_dict['session_id'] = session_id
        SESSION_DB[session_id] = session_dict
        
        # Generate the PDF
        generator = ReportGenerator()
        pdf_buffer = generator.generate_pdf(session_dict)
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=ECG_Report_{session_id}.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Generation failed: {str(e)}")


@router.get("/session/{session_id}/json")
async def export_session_json(session_id: str, authenticated: bool = Depends(verify_report_access)):
    """
    Exports the raw structured JSON data for a given session.
    Useful for downstream analytics or interoperability (e.g. FHIR integration).
    """
    if session_id not in SESSION_DB:
        raise HTTPException(status_code=404, detail="Session not found")
        
    return JSONResponse(status_code=200, content=SESSION_DB[session_id])

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uuid

from reports.report_generator import ReportGenerator

router = APIRouter(prefix="/report", tags=["Clinical Reports"])

# In-memory store for session JSONs for the GET endpoint
# In a real app, this would be a Postgres or MongoDB database
SESSION_DB = {}

# --- Pydantic Models for Validation ---

class PatientMetadata(BaseModel):
    id: str
    age: int
    gender: str
    session_date: str

class SignalMetadata(BaseModel):
    duration_sec: float
    sampling_rate: int
    snr_before: float
    snr_after: float

class ClassDistribution(BaseModel):
    N: int
    V: int
    A: int
    L: int
    R: int

class BeatStatistics(BaseModel):
    total_beats: int
    class_distribution: ClassDistribution
    dominant_rhythm: str

class AnomalyEvent(BaseModel):
    timestamp: float
    beat_type: str
    confidence: float
    alert_message: str

class ModelMetrics(BaseModel):
    average_confidence: float
    low_confidence_beats: int

class SessionData(BaseModel):
    session_id: Optional[str] = None
    patient_metadata: PatientMetadata
    signal_metadata: SignalMetadata
    beat_statistics: BeatStatistics
    anomaly_events: List[AnomalyEvent]
    model_metrics: ModelMetrics
    waveform_b64: Optional[str] = None
    confusion_matrix_b64: Optional[str] = None


@router.post("/generate")
async def generate_report(data: SessionData):
    """
    Generates a professional clinical PDF report from ECG session data.
    """
    try:
        # Assign an ID if none exists and store for the GET endpoint
        session_id = data.session_id or str(uuid.uuid4())
        session_dict = data.dict()
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
async def export_session_json(session_id: str):
    """
    Exports the raw structured JSON data for a given session.
    Useful for downstream analytics or interoperability (e.g. FHIR integration).
    """
    if session_id not in SESSION_DB:
        raise HTTPException(status_code=404, detail="Session not found")
        
    return JSONResponse(status_code=200, content=SESSION_DB[session_id])

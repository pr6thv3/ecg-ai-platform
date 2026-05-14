from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import List

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    MODEL_PATH: Path = Path("models/ecg_cnn.onnx")
    MODEL_SHA256: str | None = None
    ALLOWED_ORIGINS: str = "http://localhost:3000"
    USE_ONNX: bool = True
    LOG_LEVEL: str = "INFO"
    PROMETHEUS_MULTIPROC_DIR: str | None = None
    METRICS_AUTH_TOKEN: str | None = None
    REPORT_AUTH_TOKEN: str | None = None
    MAX_BEAT_WINDOW_LENGTH: int = 360
    MAX_REPORT_EVENTS: int = 500
    MAX_REPORT_IMAGE_BYTES: int = 1_000_000
    ALLOWED_MITBIH_RECORDS: str = "100,101,103,105,111"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

    @property
    def allowed_mitbih_records_set(self) -> set[str]:
        return {record.strip() for record in self.ALLOWED_MITBIH_RECORDS.split(",") if record.strip()}

settings = Settings()

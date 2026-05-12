from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List

class Settings(BaseSettings):
    MODEL_PATH: Path = Path("/opt/model/ecg_cnn.onnx")
    ALLOWED_ORIGINS: str = "http://localhost:3000"
    USE_ONNX: bool = True
    LOG_LEVEL: str = "INFO"
    PROMETHEUS_MULTIPROC_DIR: str | None = None
    METRICS_AUTH_TOKEN: str | None = None

    class Config:
        env_file = ".env"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]

settings = Settings()

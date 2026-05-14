from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import resolve_path


class InferenceLogger:
    def __init__(self, path: str | Path):
        self.path = resolve_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fields = [
            "timestamp_utc",
            "runtime",
            "input_filename",
            "input_samples",
            "input_length",
            "window_count",
            "prediction",
            "confidence",
            "processing_time_ms",
            "latency_ms",
            "status",
            "error_flag",
            "error",
            "source_mode",
        ]
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.fields).writeheader()

    def log(self, payload: dict[str, Any]) -> None:
        row = {field: payload.get(field, "") for field in self.fields}
        row["timestamp_utc"] = row["timestamp_utc"] or datetime.now(timezone.utc).isoformat()
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.fields).writerow(row)

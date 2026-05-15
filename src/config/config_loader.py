from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised when a YAML config is missing required project settings."""


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path = "configs/default.yaml", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = resolve_path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if overrides:
        config = _merge_dict(config, overrides)

    _apply_env_overrides(config)
    _validate_config(config, path)
    config["_config_path"] = str(path)
    config["_repo_root"] = str(REPO_ROOT)
    return config


def _validate_config(config: dict[str, Any], path: Path) -> None:
    required_sections = [
        "dataset",
        "preprocessing",
        "model",
        "training",
        "artifacts",
        "reports",
    ]
    missing = [section for section in required_sections if section not in config]
    if missing:
        raise ConfigError(f"{path} is missing required section(s): {', '.join(missing)}")

    class_names = config["model"].get("class_names", [])
    class_mapping = config["dataset"].get("class_mapping", {})
    if not class_names or len(class_names) != len(class_mapping):
        raise ConfigError("model.class_names and dataset.class_mapping must describe the same classes")

    ratios = config["dataset"].get("split", {})
    total = sum(float(ratios.get(k, 0.0)) for k in ("train_ratio", "val_ratio", "test_ratio"))
    if abs(total - 1.0) > 1e-6:
        raise ConfigError("dataset.split train/val/test ratios must sum to 1.0")

    model_type = config["model"].get("type", "baseline_cnn")
    if model_type not in {"baseline_cnn", "resnet1d", "inceptiontime", "cnn_lstm"}:
        raise ConfigError("model.type must be one of baseline_cnn, resnet1d, inceptiontime, cnn_lstm")

    normalization = str(config["preprocessing"].get("normalization", "maxabs"))
    if normalization not in {"maxabs", "zscore", "robust_zscore", "none"}:
        raise ConfigError("preprocessing.normalization must be one of maxabs, zscore, robust_zscore, none")

    loss = str(config["training"].get("loss", "cross_entropy"))
    if loss not in {"cross_entropy", "focal"}:
        raise ConfigError("training.loss must be cross_entropy or focal")


def resolve_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else REPO_ROOT / raw


def ensure_config_dirs(config: dict[str, Any]) -> None:
    for section_name in ("artifacts", "reports"):
        section = config.get(section_name, {})
        for key, value in section.items():
            path = resolve_path(value)
            target = path.parent if path.suffix else path
            target.mkdir(parents=True, exist_ok=True)


def _apply_env_overrides(config: dict[str, Any]) -> None:
    bool_envs = {
        "ECG_ALLOW_SYNTHETIC_FALLBACK": ("dataset", "allow_synthetic_fallback"),
        "ECG_ALLOW_MODEL_FALLBACK": ("model", "allow_fallback_checkpoint"),
        "ECG_ALLOW_MISSING_CHECKPOINT": ("model", "allow_missing_checkpoint"),
    }
    for env_name, (section, key) in bool_envs.items():
        value = os.getenv(env_name)
        if value is not None:
            config.setdefault(section, {})[key] = value.strip().lower() in {"1", "true", "yes", "on"}

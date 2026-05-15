from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import ensure_config_dirs, load_config, resolve_path
from src.evaluation import evaluate_model, tune_thresholds
from src.training import train_model
from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled MIT-BIH model-quality experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--cap", type=int, default=2500)
    args = parser.parse_args()

    base_config = load_config(args.config)
    ensure_config_dirs(base_config)
    summary = run_quality_experiments(base_config, epochs=args.epochs, cap=args.cap)
    print(json.dumps(summary, indent=2))


def run_quality_experiments(config: dict[str, Any], epochs: int, cap: int) -> dict[str, Any]:
    out_dir = resolve_path(config["reports"].get("experiments_dir", "reports/experiments"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    baseline_metrics = evaluate_model(config, checkpoint=config["model"]["checkpoint"])
    rows.append(_metrics_row("baseline_current_checkpoint", "baseline", config, baseline_metrics, trained=False))

    experiments = [
        (
            "focal_sampling_baseline_cnn",
            {
                "model": {"type": "baseline_cnn", "dropout": 0.35},
                "training": {
                    "loss": "focal",
                    "focal_gamma": 2.0,
                    "learning_rate": 0.0007,
                    "max_class_weight": 12.0,
                    "sampler_weight_power": 1.0,
                    "weighted_sampler": True,
                    "augmentation": _morphology_preserving_augmentation(),
                },
            },
            "Improved loss/sampling experiment: focal loss, stronger weighted sampling, conservative augmentation.",
        ),
        (
            "inceptiontime_architecture",
            {
                "model": {"type": "inceptiontime", "dropout": 0.35},
                "training": {
                    "loss": "focal",
                    "focal_gamma": 2.0,
                    "learning_rate": 0.0007,
                    "max_class_weight": 12.0,
                    "sampler_weight_power": 1.0,
                    "weighted_sampler": True,
                    "augmentation": _morphology_preserving_augmentation(),
                },
            },
            "Architecture experiment: InceptionTime-style multi-scale kernels under the same focal/sampling settings.",
        ),
        (
            "zscore_preprocessing_baseline_cnn",
            {
                "model": {"type": "baseline_cnn", "dropout": 0.35},
                "preprocessing": {"normalization": "zscore"},
                "training": {
                    "loss": "focal",
                    "focal_gamma": 2.0,
                    "learning_rate": 0.0007,
                    "max_class_weight": 12.0,
                    "sampler_weight_power": 1.0,
                    "weighted_sampler": True,
                    "augmentation": _morphology_preserving_augmentation(),
                },
            },
            "Preprocessing experiment: per-beat z-score normalization while preserving annotation-centered segmentation.",
        ),
    ]

    for name, overrides, description in experiments:
        experiment_config = _experiment_config(config, name, overrides, epochs=epochs, cap=cap)
        print(f"\nRunning quality experiment: {name}\n{description}\n")
        train_summary = train_model(experiment_config)
        threshold_summary = tune_thresholds(experiment_config, checkpoint=experiment_config["model"]["checkpoint"])
        metrics = evaluate_model(experiment_config, checkpoint=experiment_config["model"]["checkpoint"])
        row = _metrics_row(name, description, experiment_config, metrics, trained=True)
        row.update(
            {
                "best_val_macro_f1": train_summary["best_val_macro_f1"],
                "epochs_ran": train_summary["epochs_ran"],
                "threshold_validation_macro_f1": threshold_summary["validation_macro_f1"],
                "argmax_validation_macro_f1": threshold_summary["argmax_validation_macro_f1"],
            }
        )
        rows.append(row)
        write_json(out_dir / name / "experiment_summary.json", row)

    frame = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    frame.to_csv(out_dir / "experiment_results.csv", index=False)
    best = frame.iloc[0].to_dict() if not frame.empty else {}
    summary = {
        "selection_metric": "macro_f1",
        "epochs": epochs,
        "max_train_samples_per_class": cap,
        "baseline_macro_f1": float(baseline_metrics["macro_f1"]),
        "best_experiment": best,
        "rows": rows,
        "note": "Experiments are controlled local runs, not clinical validation.",
        "disclaimer": config["project"]["disclaimer"],
    }
    write_json(out_dir / "experiment_summary.json", summary)
    _write_markdown(out_dir / "experiment_summary.md", summary)
    return summary


def _experiment_config(
    base: dict[str, Any],
    name: str,
    overrides: dict[str, Any],
    epochs: int,
    cap: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    _deep_update(config, overrides)
    config["model"]["checkpoint"] = f"artifacts/models/experiments/{name}.pt"
    config["model"]["threshold_path"] = f"artifacts/evaluation/experiments/{name}_thresholds.json"
    config["training"]["epochs"] = int(epochs)
    config["training"]["patience"] = min(5, int(config["training"].get("patience", 5)))
    config["training"]["max_train_samples_per_class"] = int(cap)
    config["training"]["uncapped"] = False
    config["artifacts"]["metrics_dir"] = f"artifacts/metrics/experiments/{name}"
    config["artifacts"]["split_manifest"] = f"artifacts/metrics/experiments/{name}/split_manifest.json"
    config["reports"]["evaluation_dir"] = f"reports/experiments/{name}/evaluation"
    config["reports"]["error_analysis_dir"] = f"reports/experiments/{name}/error_analysis"
    return config


def _deep_update(target: dict[str, Any], overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _morphology_preserving_augmentation() -> dict[str, Any]:
    return {
        "enabled": True,
        "gaussian_noise_std": 0.01,
        "amplitude_scale_min": 0.95,
        "amplitude_scale_max": 1.05,
        "baseline_drift_std": 0.0,
        "time_shift_max": 4,
    }


def _metrics_row(
    name: str,
    description: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    trained: bool,
) -> dict[str, Any]:
    class_report = metrics["classification_report"]
    row = {
        "experiment": name,
        "description": description,
        "trained": trained,
        "model_type": config["model"].get("type"),
        "normalization": config["preprocessing"].get("normalization", "maxabs"),
        "loss": config["training"].get("loss", "cross_entropy"),
        "checkpoint": config["model"].get("checkpoint"),
        "threshold_path": config["model"].get("threshold_path"),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "roc_auc_ovr_macro": metrics.get("roc_auc_ovr_macro"),
    }
    for class_name in config["model"]["class_names"]:
        row[f"{class_name}_precision"] = class_report[class_name]["precision"]
        row[f"{class_name}_recall"] = class_report[class_name]["recall"]
        row[f"{class_name}_f1"] = class_report[class_name]["f1-score"]
    return row


def _write_markdown(path, summary: dict[str, Any]) -> None:
    best = summary.get("best_experiment", {})
    lines = [
        "# MIT-BIH Model Quality Experiments",
        "",
        "These are controlled research experiments for model debugging only. They are not clinical validation.",
        "",
        f"- Selection metric: `{summary['selection_metric']}`",
        f"- Baseline macro F1: `{summary['baseline_macro_f1']:.4f}`",
        f"- Best experiment: `{best.get('experiment', 'n/a')}`",
        f"- Best macro F1: `{float(best.get('macro_f1', 0.0)):.4f}`",
        "",
        "## Experiment Ranking",
        "",
    ]
    for row in sorted(summary["rows"], key=lambda item: item["macro_f1"], reverse=True):
        lines.append(
            f"- `{row['experiment']}`: macro F1 `{row['macro_f1']:.4f}`, "
            f"A F1 `{row.get('A_f1', 0.0):.4f}`, L F1 `{row.get('L_f1', 0.0):.4f}`, R F1 `{row.get('R_f1', 0.0):.4f}`"
        )
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

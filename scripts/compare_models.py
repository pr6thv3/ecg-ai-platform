from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd

from src.config import ensure_config_dirs, load_config, resolve_path
from src.evaluation import evaluate_model
from src.training import train_model
from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate selected ECG model architectures and compare macro F1.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", default="baseline_cnn,resnet1d,inceptiontime,cnn_lstm")
    parser.add_argument("--single-model", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--row-output", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    base_config = load_config(args.config)
    ensure_config_dirs(base_config)
    model_types = [item.strip() for item in args.models.split(",") if item.strip()]
    out_dir = resolve_path(base_config["reports"].get("experiments_dir", "reports/experiments")) / "model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.single_model:
        if len(model_types) != 1:
            raise ValueError("--single-model expects exactly one model type")
        row = _run_single_model(base_config, model_types[0])
        if args.row_output:
            write_json(args.row_output, row)
        print(json.dumps(row, indent=2))
        return

    rows = []
    for model_type in model_types:
        row_path = out_dir / f"{model_type}_row.json"
        command = [
            sys.executable,
            "-m",
            "scripts.compare_models",
            "--config",
            args.config,
            "--models",
            model_type,
            "--single-model",
            "--row-output",
            str(row_path),
        ]
        print(f"Training comparison model in isolated process: {model_type}")
        subprocess.run(command, check=True)
        with row_path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))

    frame = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    frame.to_csv(out_dir / "comparison_metrics.csv", index=False)
    _plot_comparison(out_dir / "comparison_plot.png", frame)
    summary = {
        "best_model_type": frame.iloc[0]["model_type"] if not frame.empty else None,
        "selection_metric": "macro_f1",
        "training_capped": bool(base_config["training"].get("max_train_samples_per_class")) and not bool(base_config["training"].get("uncapped")),
        "limited_budget": True,
        "note": "Each architecture is trained in an isolated subprocess using compare_epochs and compare_max_train_samples_per_class from config.",
        "rows": rows,
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def _release_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _run_single_model(base_config: dict, model_type: str) -> dict:
    config = _comparison_config(base_config, model_type)
    print(f"Training comparison model: {model_type}")
    train_summary = train_model(config)
    metrics = evaluate_model(config, checkpoint=config["model"]["checkpoint"])
    row = {
        "model_type": model_type,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "roc_auc_ovr_macro": metrics.get("roc_auc_ovr_macro"),
        "best_val_macro_f1": train_summary["best_val_macro_f1"],
        "epochs_ran": train_summary["epochs_ran"],
        "training_capped": bool(config["training"].get("max_train_samples_per_class")) and not bool(config["training"].get("uncapped")),
        "limited_budget": True,
        "checkpoint": config["model"]["checkpoint"],
    }
    for class_name, class_metrics in metrics["classification_report"].items():
        if class_name in config["model"]["class_names"]:
            row[f"{class_name}_f1"] = class_metrics["f1-score"]
            row[f"{class_name}_recall"] = class_metrics["recall"]
    _release_memory()
    return row


def _comparison_config(config: dict, model_type: str) -> dict:
    checkpoint = f"artifacts/models/model_comparison/{model_type}_best_model.pt"
    copied = {
        **config,
        "model": {
            **config["model"],
            "type": model_type,
            "checkpoint": checkpoint,
            "threshold_path": "",
        },
        "training": {
            **config["training"],
            "epochs": int(config["training"].get("compare_epochs", 3)),
            "patience": min(3, int(config["training"].get("patience", 3))),
            "batch_size": min(128, int(config["training"].get("batch_size", 512))),
            "eval_batch_size": 256,
            "max_train_samples_per_class": int(config["training"].get("compare_max_train_samples_per_class", 1000)),
        },
        "inference": {
            **config["inference"],
            "batch_size": 256,
        },
        "artifacts": {
            **config["artifacts"],
            "metrics_dir": f"artifacts/metrics/model_comparison/{model_type}",
            "split_manifest": f"artifacts/metrics/model_comparison/{model_type}/split_manifest.json",
        },
        "reports": {
            **config["reports"],
            "evaluation_dir": f"reports/experiments/model_comparison/{model_type}/evaluation",
        },
    }
    return copied


def _plot_comparison(path, frame: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(frame["model_type"], frame["macro_f1"], color="#2563eb")
    plt.xlabel("Model type")
    plt.ylabel("Macro F1")
    plt.title("Model Comparison")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


if __name__ == "__main__":
    main()

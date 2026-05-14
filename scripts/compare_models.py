from __future__ import annotations

import argparse
import json
import shutil

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
    args = parser.parse_args()

    base_config = load_config(args.config)
    ensure_config_dirs(base_config)
    model_types = [item.strip() for item in args.models.split(",") if item.strip()]
    rows = []
    out_dir = resolve_path(base_config["artifacts"].get("evaluation_dir", "artifacts/evaluation")) / "model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_type in model_types:
        config = _comparison_config(base_config, model_type)
        print(f"Training comparison model: {model_type}")
        train_summary = train_model(config)
        metrics = evaluate_model(config, checkpoint=config["model"]["checkpoint"])
        checkpoint = resolve_path(config["model"]["checkpoint"])
        archived = out_dir / f"{model_type}_best_model.pt"
        if checkpoint.exists():
            shutil.copy2(checkpoint, archived)
        row = {
            "model_type": model_type,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "best_val_macro_f1": train_summary["best_val_macro_f1"],
            "epochs_ran": train_summary["epochs_ran"],
            "training_capped": bool(config["training"].get("max_train_samples_per_class")) and not bool(config["training"].get("uncapped")),
            "checkpoint": str(archived),
        }
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    frame.to_csv(out_dir / "comparison_metrics.csv", index=False)
    _plot_comparison(out_dir / "comparison_plot.png", frame)
    summary = {
        "best_model_type": frame.iloc[0]["model_type"] if not frame.empty else None,
        "selection_metric": "macro_f1",
        "training_capped": bool(base_config["training"].get("max_train_samples_per_class")) and not bool(base_config["training"].get("uncapped")),
        "rows": rows,
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def _comparison_config(config: dict, model_type: str) -> dict:
    copied = {
        **config,
        "model": {**config["model"], "type": model_type, "threshold_path": ""},
        "training": {
            **config["training"],
            "epochs": int(config["training"].get("compare_epochs", 3)),
            "patience": min(3, int(config["training"].get("patience", 3))),
            "max_train_samples_per_class": int(config["training"].get("compare_max_train_samples_per_class", 1000)),
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

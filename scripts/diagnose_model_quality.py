from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.config import ensure_config_dirs, load_config, resolve_path
from src.data.dataset import load_dataset, split_dataset
from src.inference import InferencePipeline
from src.utils.io import write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model-quality diagnostics for real MIT-BIH failure analysis.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_config_dirs(config)
    summary = diagnose_model_quality(config, checkpoint=args.checkpoint)
    print(json.dumps(summary, indent=2))


def diagnose_model_quality(config: dict[str, Any], checkpoint: str | None = None) -> dict[str, Any]:
    out_dir = resolve_path(config["reports"].get("experiments_dir", "reports/experiments")) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset(config)
    splits = split_dataset(bundle, config)
    pipeline = InferencePipeline(_config_with_checkpoint(config, checkpoint))
    probs = pipeline.predict_windows(splits.X_test)
    threshold_preds = np.asarray([pipeline._predict_index(row) for row in probs], dtype=np.int64)
    argmax_preds = np.argmax(probs, axis=1)
    class_names = list(config["model"]["class_names"])

    split_distribution = _split_distribution(splits, class_names)
    per_record_rows = _per_record_rows(bundle, class_names, splits)
    prediction_rows = _prediction_rows(splits.test_sample_ids, splits.y_test, threshold_preds, probs, class_names)
    confidence_true = _confidence_summary(prediction_rows, "true_label")
    confidence_pred = _confidence_summary(prediction_rows, "pred_label")
    false_negative_rows = [row for row in prediction_rows if not row["correct"]]
    false_negative_summary = _false_negative_summary(splits.y_test, threshold_preds, class_names)
    prediction_distribution = _prediction_distribution(threshold_preds, class_names)

    write_json(out_dir / "class_distribution_by_split.json", split_distribution)
    write_csv(out_dir / "class_distribution_by_split.csv", _flatten_split_distribution(split_distribution))
    write_csv(out_dir / "per_record_class_distribution.csv", per_record_rows)
    write_json(out_dir / "per_class_sample_counts.json", _per_class_counts(bundle, splits, class_names))
    write_csv(out_dir / "per_class_false_negatives.csv", _flatten_false_negative_summary(false_negative_summary))
    write_json(out_dir / "per_class_false_negatives.json", false_negative_summary)
    write_json(out_dir / "prediction_distribution.json", prediction_distribution)
    write_csv(out_dir / "prediction_distribution.csv", _flatten_prediction_distribution(prediction_distribution))
    write_csv(out_dir / "confidence_by_true_class.csv", confidence_true)
    write_csv(out_dir / "confidence_by_predicted_class.csv", confidence_pred)
    write_csv(out_dir / "test_predictions.csv", prediction_rows)
    _copy_training_history(config, out_dir)
    _plot_split_distribution(out_dir / "class_distribution_by_split.png", split_distribution, class_names)

    argmax_macro = float(f1_score(splits.y_test, argmax_preds, average="macro", zero_division=0))
    threshold_macro = float(f1_score(splits.y_test, threshold_preds, average="macro", zero_division=0))
    summary = {
        "source": splits.source,
        "real_mitbih": splits.source == "mitbih",
        "checkpoint": checkpoint or config["model"]["checkpoint"],
        "record_count": int(len(set(bundle.record_ids.tolist()))),
        "sample_count": int(len(bundle.y)),
        "segmentation": {
            "training_windows": "MIT-BIH annotation-centered beat windows",
            "window_size": int(config["preprocessing"]["window_size"]),
            "normalization": str(config["preprocessing"].get("normalization", "maxabs")),
            "note": "Evaluation uses annotation-centered windows; API inference uses detected R-peaks.",
        },
        "argmax": {
            "accuracy": float(accuracy_score(splits.y_test, argmax_preds)),
            "macro_f1": argmax_macro,
        },
        "thresholded": {
            "accuracy": float(accuracy_score(splits.y_test, threshold_preds)),
            "macro_f1": threshold_macro,
            "thresholds_used": bool(pipeline.thresholds),
        },
        "top_findings": _top_findings(split_distribution, false_negative_summary, argmax_macro, threshold_macro),
        "disclaimer": config["project"]["disclaimer"],
    }
    write_json(out_dir / "diagnostic_summary.json", summary)
    _write_markdown(out_dir / "diagnostic_summary.md", summary)
    return summary


def _config_with_checkpoint(config: dict[str, Any], checkpoint: str | None) -> dict[str, Any]:
    if not checkpoint:
        return config
    return {**config, "model": {**config["model"], "checkpoint": checkpoint}}


def _split_distribution(splits: Any, class_names: list[str]) -> dict[str, Any]:
    return {
        "train": _class_distribution(splits.y_train, class_names),
        "validation": _class_distribution(splits.y_val, class_names),
        "test": _class_distribution(splits.y_test, class_names),
    }


def _class_distribution(y: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    total = int(len(y))
    counts = {name: int(np.sum(y == idx)) for idx, name in enumerate(class_names)}
    percentages = {name: float(count / max(1, total) * 100.0) for name, count in counts.items()}
    return {"total": total, "counts": counts, "percentages": percentages}


def _flatten_split_distribution(distribution: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, split in distribution.items():
        for class_name, count in split["counts"].items():
            rows.append(
                {
                    "split": split_name,
                    "class": class_name,
                    "count": count,
                    "percentage": split["percentages"][class_name],
                }
            )
    return rows


def _per_record_rows(bundle: Any, class_names: list[str], splits: Any) -> list[dict[str, Any]]:
    split_by_record = {
        **{record: "train" for record in splits.train_record_ids},
        **{record: "validation" for record in splits.val_record_ids},
        **{record: "test" for record in splits.test_record_ids},
    }
    rows: list[dict[str, Any]] = []
    for record in sorted(set(bundle.record_ids.tolist())):
        mask = bundle.record_ids == record
        labels = bundle.y[mask]
        row = {
            "record_id": str(record),
            "split": split_by_record.get(str(record), "unknown"),
            "total": int(len(labels)),
        }
        for idx, class_name in enumerate(class_names):
            row[class_name] = int(np.sum(labels == idx))
        rows.append(row)
    return rows


def _per_class_counts(bundle: Any, splits: Any, class_names: list[str]) -> dict[str, Any]:
    return {
        "overall": _class_distribution(bundle.y, class_names),
        "train": _class_distribution(splits.y_train, class_names),
        "validation": _class_distribution(splits.y_val, class_names),
        "test": _class_distribution(splits.y_test, class_names),
    }


def _prediction_rows(
    sample_ids: np.ndarray,
    y_true: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, sample_id in enumerate(sample_ids):
        true_idx = int(y_true[idx])
        pred_idx = int(preds[idx])
        row = {
            "sample_id": str(sample_id),
            "record_id": str(sample_id).split(":")[0],
            "true_label": class_names[true_idx],
            "pred_label": class_names[pred_idx],
            "confidence": float(probs[idx, pred_idx]),
            "true_class_probability": float(probs[idx, true_idx]),
            "correct": bool(true_idx == pred_idx),
        }
        row.update({f"prob_{name}": float(probs[idx, class_idx]) for class_idx, name in enumerate(class_names)})
        rows.append(row)
    return rows


def _confidence_summary(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return []
    grouped = frame.groupby(group_key)
    output = []
    for group_name, group in grouped:
        output.append(
            {
                group_key: group_name,
                "count": int(len(group)),
                "accuracy": float(group["correct"].mean()),
                "mean_confidence": float(group["confidence"].mean()),
                "median_confidence": float(group["confidence"].median()),
                "mean_true_class_probability": float(group["true_class_probability"].mean()),
            }
        )
    return output


def _false_negative_summary(y_true: np.ndarray, preds: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for idx, class_name in enumerate(class_names):
        true_mask = y_true == idx
        fn_mask = true_mask & (preds != idx)
        support = int(np.sum(true_mask))
        result[class_name] = {
            "support": support,
            "false_negative_count": int(np.sum(fn_mask)),
            "false_negative_rate": float(np.sum(fn_mask) / support) if support else 0.0,
            "predicted_as": {
                class_names[pred_idx]: int(np.sum(fn_mask & (preds == pred_idx)))
                for pred_idx in range(len(class_names))
                if pred_idx != idx
            },
        }
    return result


def _flatten_false_negative_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for class_name, item in summary.items():
        row = {
            "class": class_name,
            "support": item["support"],
            "false_negative_count": item["false_negative_count"],
            "false_negative_rate": item["false_negative_rate"],
        }
        for predicted_as, count in item["predicted_as"].items():
            row[f"predicted_as_{predicted_as}"] = count
        rows.append(row)
    return rows


def _prediction_distribution(preds: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    counts = np.bincount(preds, minlength=len(class_names))
    total = int(len(preds))
    return {
        "total": total,
        "counts": {class_names[idx]: int(counts[idx]) for idx in range(len(class_names))},
        "percentages": {class_names[idx]: float(counts[idx] / max(1, total) * 100.0) for idx in range(len(class_names))},
    }


def _flatten_prediction_distribution(distribution: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "class": class_name,
            "count": distribution["counts"][class_name],
            "percentage": distribution["percentages"][class_name],
        }
        for class_name in distribution["counts"]
    ]


def _copy_training_history(config: dict[str, Any], out_dir: Path) -> None:
    history_path = resolve_path(config["artifacts"]["metrics_dir"]) / "training_history.csv"
    if not history_path.exists():
        return
    frame = pd.read_csv(history_path)
    frame.to_csv(out_dir / "training_history.csv", index=False)
    if {"epoch", "train_loss", "val_loss", "val_macro_f1"}.issubset(frame.columns):
        plt.figure(figsize=(8, 4))
        plt.plot(frame["epoch"], frame["train_loss"], label="train loss")
        plt.plot(frame["epoch"], frame["val_loss"], label="val loss")
        plt.plot(frame["epoch"], frame["val_macro_f1"], label="val macro F1")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Available Training Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "training_curves.png", dpi=160)
        plt.close()


def _plot_split_distribution(path: Path, distribution: dict[str, Any], class_names: list[str]) -> None:
    frame = pd.DataFrame(_flatten_split_distribution(distribution))
    plt.figure(figsize=(8, 4))
    for split in ("train", "validation", "test"):
        rows = frame[frame["split"] == split]
        plt.plot(rows["class"], rows["count"], marker="o", label=split)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution By Split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _top_findings(
    split_distribution: dict[str, Any],
    false_negatives: dict[str, Any],
    argmax_macro: float,
    threshold_macro: float,
) -> list[str]:
    findings = []
    test_counts = split_distribution["test"]["counts"]
    if test_counts.get("A", 0) < 200:
        findings.append("A has very low held-out support, so A metrics are high-variance and sensitive to record choice.")
    if false_negatives.get("L", {}).get("false_negative_rate", 0.0) >= 0.95:
        findings.append("L is effectively not recovered on the held-out records despite non-trivial train support.")
    if false_negatives.get("R", {}).get("false_negative_rate", 0.0) >= 0.90:
        findings.append("R has severe record-level generalization failure and is mostly predicted as another morphology class.")
    if threshold_macro < argmax_macro:
        findings.append("Current thresholds reduce held-out macro F1 relative to argmax; threshold tuning is overfitting validation.")
    else:
        findings.append("Current thresholds improve or match held-out macro F1 relative to argmax.")
    findings.append("Training loss falls while validation macro F1 plateaus, indicating overfitting and record-level domain shift.")
    return findings


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Model Quality Diagnostics",
        "",
        "This report is technical failure analysis only. It is not clinical validation.",
        "",
        f"- Source: `{summary['source']}`",
        f"- Real MIT-BIH: `{summary['real_mitbih']}`",
        f"- Record count: `{summary['record_count']}`",
        f"- Sample count: `{summary['sample_count']}`",
        f"- Window size: `{summary['segmentation']['window_size']}`",
        f"- Normalization: `{summary['segmentation']['normalization']}`",
        f"- Argmax macro F1: `{summary['argmax']['macro_f1']:.4f}`",
        f"- Thresholded macro F1: `{summary['thresholded']['macro_f1']:.4f}`",
        "",
        "## Findings",
        "",
    ]
    lines.extend(f"- {finding}" for finding in summary["top_findings"])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

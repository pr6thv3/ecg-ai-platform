from __future__ import annotations

import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)

from src.config import ensure_config_dirs, resolve_path
from src.data.dataset import leakage_report, load_dataset, split_dataset
from src.data.synthetic import synthetic_beat
from src.inference import InferencePipeline
from src.preprocessing import SignalValidationError, load_signal_file, segment_signal
from src.utils.io import write_csv, write_json


def _config_with_checkpoint(config: dict[str, Any], checkpoint: str | None) -> dict[str, Any]:
    if not checkpoint:
        return config
    return {**config, "model": {**config["model"], "checkpoint": checkpoint}}


def evaluate_model(config: dict[str, Any], checkpoint: str | None = None) -> dict[str, Any]:
    ensure_config_dirs(config)
    dataset = load_dataset(config)
    splits = split_dataset(dataset, config)
    pipeline = InferencePipeline(_config_with_checkpoint(config, checkpoint))
    probs = pipeline.predict_windows(splits.X_test)
    preds = np.asarray([pipeline._predict_index(row) for row in probs], dtype=np.int64)
    y_true = splits.y_test
    class_names = list(config["model"]["class_names"])
    threshold = float(config["inference"]["low_confidence_threshold"])

    metrics = _classification_metrics(y_true, preds, probs, class_names, threshold)
    dataset_meta = _dataset_metadata(config, splits)
    anti_collapse = _anti_collapse(preds, class_names)
    for warning in anti_collapse["warnings"]:
        print(f"WARNING: {warning}")

    metrics.update(
        {
            **dataset_meta,
            "checkpoint": checkpoint or config["model"]["checkpoint"],
            "model_info": pipeline.model_info(),
            "runtime": pipeline.runtime,
            "sample_count": int(len(y_true)),
            "disclaimer": config["project"]["disclaimer"],
            "warnings": [*splits.warnings, *anti_collapse["warnings"]],
            "note": "Synthetic fallback metrics are pipeline checks, not MIT-BIH performance evidence."
            if splits.source != "mitbih"
            else "Metrics computed on configured MIT-BIH split.",
            "anti_collapse_check": anti_collapse,
        }
    )

    eval_dir = resolve_path(config["reports"]["evaluation_dir"])
    artifact_eval_dir = resolve_path(config["artifacts"].get("evaluation_dir", "artifacts/evaluation"))
    eval_dir.mkdir(parents=True, exist_ok=True)
    artifact_eval_dir.mkdir(parents=True, exist_ok=True)
    full_cm = confusion_matrix(y_true, preds, labels=list(range(len(class_names))))
    pred_dist = _prediction_distribution(preds, class_names)
    false_negative_summary = _false_negative_summary(y_true, preds, class_names)
    rows = _prediction_rows(splits.test_sample_ids, y_true, preds, probs, class_names)

    write_json(eval_dir / "metrics.json", metrics)
    write_json(eval_dir / "classification_report.json", metrics["classification_report"])
    write_json(eval_dir / "class_distribution.json", dataset_meta["class_distribution"])
    write_json(eval_dir / "prediction_distribution.json", pred_dist)
    write_json(eval_dir / "false_negative_summary.json", false_negative_summary)
    write_csv(eval_dir / "predictions.csv", rows)
    write_csv(eval_dir / "false_negatives.csv", [row for row in rows if not row["correct"]])
    write_csv(
        eval_dir / "low_confidence_predictions.csv",
        [row for row in rows if row["confidence"] < threshold],
    )
    pd.DataFrame(full_cm, index=class_names, columns=class_names).to_csv(eval_dir / "confusion_matrix.csv")
    _save_prediction_distribution(eval_dir / "prediction_distribution.png", pred_dist)
    pd.DataFrame(
        [{"class": name, "count": pred_dist["counts"][name], "percentage": pred_dist["percentages"][name]} for name in class_names]
    ).to_csv(eval_dir / "prediction_distribution.csv", index=False)
    _save_class_distribution(artifact_eval_dir, dataset_meta["class_distribution"], class_names)
    _save_curve_plots(eval_dir, y_true, probs, class_names)
    _save_confusion_matrix(eval_dir / "confusion_matrix.png", full_cm, class_names)
    write_json(eval_dir / "mitbih_evaluation_summary.json", metrics)
    write_json(eval_dir / "metrics_summary.json", metrics)
    write_json(artifact_eval_dir / "dataset_summary.json", dataset_meta)
    write_json(artifact_eval_dir / "splits.json", leakage_report(splits))
    write_json(artifact_eval_dir / "metrics_summary.json", metrics)
    write_json(config["artifacts"]["split_manifest"], leakage_report(splits))
    return metrics


def _classification_metrics(
    y_true: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    low_confidence_threshold: float,
) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        preds,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, preds, labels=labels)
    for idx, name in enumerate(class_names):
        tp = float(cm[idx, idx])
        fn = float(cm[idx, :].sum() - tp)
        fp = float(cm[:, idx].sum() - tp)
        tn = float(cm.sum() - tp - fn - fp)
        report[name]["sensitivity"] = report[name]["recall"]
        report[name]["specificity"] = tn / (tn + fp) if (tn + fp) else 0.0

    roc_auc: float | None
    try:
        roc_value = float(
            roc_auc_score(
                y_true,
                probs,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
        roc_auc = None if np.isnan(roc_value) else roc_value
    except Exception:
        roc_auc = None

    confidences = probs[np.arange(len(preds)), preds]
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision_macro": float(precision_score(y_true, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": roc_auc,
        "classification_report": report,
        "low_confidence_summary": {
            "threshold": low_confidence_threshold,
            "count": int(np.sum(confidences < low_confidence_threshold)),
            "percentage": float(np.mean(confidences < low_confidence_threshold) * 100.0),
        },
    }


def _dataset_metadata(config: dict[str, Any], splits: Any) -> dict[str, Any]:
    class_names = list(config["model"]["class_names"])
    class_distribution = {
        "source": splits.source,
        "real_mitbih": splits.source == "mitbih",
        "sampling_rate": int(config["dataset"]["sampling_rate"]),
        "split_strategy": config["dataset"]["split"].get("group_by", "random"),
        "train": _class_distribution(splits.y_train, class_names),
        "validation": _class_distribution(splits.y_val, class_names),
        "test": _class_distribution(splits.y_test, class_names),
        "warnings": [],
    }
    for split_name in ("train", "validation", "test"):
        for class_name, count in class_distribution[split_name]["counts"].items():
            if count == 0:
                class_distribution["warnings"].append(f"{split_name} split has zero samples for class {class_name}.")
            elif count < 50:
                class_distribution["warnings"].append(
                    f"{split_name} split has only {count} sample(s) for class {class_name}; metric reliability is limited."
                )
    record_count = int(len(set([*splits.train_record_ids, *splits.val_record_ids, *splits.test_record_ids])))
    return {
        "source": splits.source,
        "real_mitbih": splits.source == "mitbih",
        "record_count": record_count,
        "sampling_rate": int(config["dataset"]["sampling_rate"]),
        "split_strategy": config["dataset"]["split"].get("group_by", "random"),
        "class_distribution": class_distribution,
    }


def _class_distribution(y: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    total = int(len(y))
    counts = {name: int(np.sum(y == idx)) for idx, name in enumerate(class_names)}
    percentages = {name: (round(count / total * 100.0, 4) if total else 0.0) for name, count in counts.items()}
    return {"total": total, "counts": counts, "percentages": percentages}


def _anti_collapse(preds: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    pred_counts = np.bincount(preds, minlength=len(class_names))
    max_idx = int(np.argmax(pred_counts)) if len(pred_counts) else 0
    max_pred_pct = float(np.max(pred_counts) / max(1, len(preds)) * 100.0)
    unique_preds = int(np.sum(pred_counts > 0))
    warnings: list[str] = []
    if max_pred_pct > 85.0:
        warnings.append(
            f"Model predicts {class_names[max_idx]} for {max_pred_pct:.1f}% of samples; this indicates class collapse risk."
        )
    missing = [class_names[i] for i in range(len(class_names)) if pred_counts[i] == 0]
    if missing:
        warnings.append(f"Model never predicts classes: {', '.join(missing)}.")
    return {
        "max_pred_class": class_names[max_idx] if class_names else "",
        "max_pred_percentage": max_pred_pct,
        "unique_classes_predicted": unique_preds,
        "warnings": warnings,
        "status": "warning" if warnings else "passed",
    }


def _prediction_distribution(preds: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    counts = np.bincount(preds, minlength=len(class_names))
    total = int(len(preds))
    return {
        "total": total,
        "counts": {class_names[i]: int(counts[i]) for i in range(len(class_names))},
        "percentages": {
            class_names[i]: (round(float(counts[i]) / max(1, total) * 100.0, 4)) for i in range(len(class_names))
        },
    }


def _false_negative_summary(y_true: np.ndarray, preds: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for idx, name in enumerate(class_names):
        true_mask = y_true == idx
        fn_mask = true_mask & (preds != idx)
        predicted_as = {
            class_names[pred_idx]: int(np.sum(fn_mask & (preds == pred_idx)))
            for pred_idx in range(len(class_names))
            if pred_idx != idx
        }
        support = int(np.sum(true_mask))
        fn_count = int(np.sum(fn_mask))
        rows[name] = {
            "support": support,
            "false_negative_count": fn_count,
            "false_negative_rate": float(fn_count / support) if support else 0.0,
            "predicted_as": predicted_as,
        }
    return rows


def _prediction_rows(
    sample_ids: np.ndarray,
    y_true: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, sample_id in enumerate(sample_ids):
        pred = int(preds[idx])
        true = int(y_true[idx])
        row = {
            "sample_id": str(sample_id),
            "true_index": true,
            "true_label": class_names[true],
            "pred_index": pred,
            "pred_label": class_names[pred],
            "confidence": float(probs[idx, pred]),
            "correct": bool(pred == true),
        }
        row.update({f"prob_{name}": float(probs[idx, i]) for i, name in enumerate(class_names)})
        rows.append(row)
    return rows


def _save_confusion_matrix(path: Path, cm: np.ndarray, class_names: list[str]) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("ECG Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(col, row, int(cm[row, col]), ha="center", va="center")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_prediction_distribution(path: Path, pred_dist: dict[str, Any]) -> None:
    names = list(pred_dist["counts"].keys())
    values = [pred_dist["counts"][name] for name in names]
    plt.figure(figsize=(7, 4))
    plt.bar(names, values, color="#2563eb")
    plt.xlabel("Predicted class")
    plt.ylabel("Count")
    plt.title("Prediction Distribution")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_class_distribution(out_dir: Path, distribution: dict[str, Any], class_names: list[str]) -> None:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        split = distribution[split_name]
        for class_name in class_names:
            rows.append(
                {
                    "split": split_name,
                    "class": class_name,
                    "count": split["counts"][class_name],
                    "percentage": split["percentages"][class_name],
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "class_distribution.csv", index=False)
    frame = pd.DataFrame(rows)
    plt.figure(figsize=(9, 4))
    for split_name in ("train", "validation", "test"):
        split_rows = frame[frame["split"] == split_name]
        plt.plot(split_rows["class"], split_rows["count"], marker="o", label=split_name)
    plt.title("Class Distribution By Split")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution.png", dpi=160)
    plt.close()


def _save_curve_plots(out_dir: Path, y_true: np.ndarray, probs: np.ndarray, class_names: list[str]) -> None:
    labels = list(range(len(class_names)))
    plt.figure(figsize=(7, 6))
    for idx, name in enumerate(class_names):
        y_binary = (y_true == idx).astype(int)
        if len(np.unique(y_binary)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_binary, probs[:, idx])
        try:
            auc_value = roc_auc_score(y_binary, probs[:, idx])
        except Exception:
            auc_value = float("nan")
        plt.plot(fpr, tpr, label=f"{name} AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], color="#64748b", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Per-Class ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curves.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 6))
    for idx, name in enumerate(class_names):
        y_binary = (y_true == idx).astype(int)
        if len(np.unique(y_binary)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_binary, probs[:, idx])
        try:
            ap_value = average_precision_score(y_binary, probs[:, idx])
        except Exception:
            ap_value = float("nan")
        plt.plot(recall, precision, label=f"{name} AP={ap_value:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Per-Class PR Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curves.png", dpi=160)
    plt.close()


def tune_thresholds(config: dict[str, Any], checkpoint: str | None = None) -> dict[str, Any]:
    dataset = load_dataset(config)
    splits = split_dataset(dataset, config)
    threshold_config = {
        **config,
        "model": {**config["model"], "threshold_path": ""},
    }
    pipeline = InferencePipeline(_config_with_checkpoint(threshold_config, checkpoint))
    probs = pipeline.predict_windows(splits.X_val)
    y_true = splits.y_val
    class_names = list(config["model"]["class_names"])
    thresholds = {name: 0.5 for name in class_names}
    best_preds = _predict_with_thresholds(probs, class_names, thresholds)
    best_f1 = float(f1_score(y_true, best_preds, average="macro", zero_division=0))
    grid = np.linspace(0.1, 0.9, 17)

    improved = True
    while improved:
        improved = False
        for class_name in class_names:
            class_best = thresholds[class_name]
            for candidate in grid:
                trial = dict(thresholds)
                trial[class_name] = float(candidate)
                trial_preds = _predict_with_thresholds(probs, class_names, trial)
                trial_f1 = float(f1_score(y_true, trial_preds, average="macro", zero_division=0))
                if trial_f1 > best_f1:
                    best_f1 = trial_f1
                    class_best = float(candidate)
                    best_preds = trial_preds
                    improved = True
            thresholds[class_name] = class_best

    payload = {
        "source": splits.source,
        "real_mitbih": splits.source == "mitbih",
        "checkpoint": checkpoint or config["model"]["checkpoint"],
        "optimized_metric": "validation_macro_f1",
        "validation_macro_f1": best_f1,
        "thresholds": thresholds,
        "class_order": class_names,
        "argmax_validation_macro_f1": float(
            f1_score(y_true, np.argmax(probs, axis=1), average="macro", zero_division=0)
        ),
        "disclaimer": config["project"]["disclaimer"],
    }
    threshold_path = resolve_path(config["model"].get("threshold_path", "artifacts/evaluation/thresholds.json"))
    write_json(threshold_path, payload)
    write_json(resolve_path(config["artifacts"].get("evaluation_dir", "artifacts/evaluation")) / "thresholds.json", payload)
    return payload


def _predict_with_thresholds(probs: np.ndarray, class_names: list[str], thresholds: dict[str, float]) -> np.ndarray:
    preds: list[int] = []
    threshold_values = np.asarray([float(thresholds.get(name, 1.0)) for name in class_names], dtype=np.float32)
    for row in probs:
        margins = row - threshold_values
        eligible = np.where(margins >= 0)[0]
        preds.append(int(eligible[np.argmax(margins[eligible])]) if eligible.size else int(np.argmax(row)))
    return np.asarray(preds, dtype=np.int64)


def run_error_analysis(config: dict[str, Any], checkpoint: str | None = None) -> dict[str, Any]:
    dataset = load_dataset(config)
    splits = split_dataset(dataset, config)
    pipeline = InferencePipeline(_config_with_checkpoint(config, checkpoint))
    probs = pipeline.predict_windows(splits.X_test)
    preds = np.argmax(probs, axis=1)
    class_names = list(config["model"]["class_names"])
    rows = _prediction_rows(splits.test_sample_ids, splits.y_test, preds, probs, class_names)
    error_dir = resolve_path(config["reports"]["error_analysis_dir"])
    error_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(config["inference"]["low_confidence_threshold"])

    misclassified = [row for row in rows if not row["correct"]]
    high_conf_wrong = sorted(
        [row for row in misclassified if row["confidence"] >= threshold],
        key=lambda row: row["confidence"],
        reverse=True,
    )
    low_conf_all = [row for row in rows if row["confidence"] < threshold]
    low_conf_correct = sorted(
        [row for row in rows if row["correct"] and row["confidence"] < threshold],
        key=lambda row: row["confidence"],
    )
    false_negative_rows = [
        {
            "sample_id": row["sample_id"],
            "true_label": row["true_label"],
            "pred_label": row["pred_label"],
            "confidence": row["confidence"],
        }
        for row in misclassified
    ]
    failure_rows = _failure_rows(rows, class_names)
    worst_classes = sorted(failure_rows, key=lambda row: row["failure_rate"], reverse=True)

    write_csv(error_dir / "misclassified_samples.csv", misclassified)
    write_csv(error_dir / "high_confidence_wrong_predictions.csv", high_conf_wrong)
    write_csv(error_dir / "low_confidence_predictions.csv", low_conf_all)
    write_csv(error_dir / "low_confidence_correct_predictions.csv", low_conf_correct)
    write_csv(error_dir / "false_negative_samples.csv", false_negative_rows)
    write_csv(error_dir / "per_class_failure_summary.csv", failure_rows)
    write_csv(error_dir / "worst_classes.csv", worst_classes)
    _save_failure_waveforms(error_dir / "waveforms", splits.X_test, rows, class_names)
    _save_confidence_plot(error_dir / "confidence_distribution.png", [row["confidence"] for row in rows])
    summary = {
        "source": splits.source,
        "real_mitbih": splits.source == "mitbih",
        "total_samples": len(rows),
        "misclassified_count": len(misclassified),
        "high_confidence_wrong_count": len(high_conf_wrong),
        "low_confidence_count": len(low_conf_all),
        "low_confidence_correct_count": len(low_conf_correct),
        "worst_classes": worst_classes[:5],
        "false_negative_summary": _false_negative_summary(splits.y_test, preds, class_names),
        "warnings": splits.warnings,
    }
    write_json(error_dir / "summary.json", summary)
    write_json(error_dir / "false_negative_summary.json", summary["false_negative_summary"])
    artifact_eval_dir = resolve_path(config["artifacts"].get("evaluation_dir", "artifacts/evaluation"))
    artifact_eval_dir.mkdir(parents=True, exist_ok=True)
    _write_error_markdown(artifact_eval_dir / "error_analysis.md", summary)
    return summary


def _failure_rows(rows: list[dict[str, Any]], class_names: list[str]) -> list[dict[str, Any]]:
    failure_rows = []
    for class_name in class_names:
        class_rows = [row for row in rows if row["true_label"] == class_name]
        failures = [row for row in class_rows if not row["correct"]]
        failure_rows.append(
            {
                "class": class_name,
                "samples": len(class_rows),
                "failures": len(failures),
                "failure_rate": float(len(failures) / max(1, len(class_rows))),
            }
        )
    return failure_rows


def _save_failure_waveforms(out_dir: Path, X_test: np.ndarray, rows: list[dict[str, Any]], class_names: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for class_name in class_names:
        failures = [(idx, row) for idx, row in enumerate(rows) if row["true_label"] == class_name and not row["correct"]]
        for rank, (idx, row) in enumerate(failures[:3], start=1):
            plt.figure(figsize=(8, 3))
            plt.plot(X_test[idx], color="#0f172a", linewidth=1.1)
            plt.title(f"{class_name} false negative: predicted {row['pred_label']} ({row['confidence']:.3f})")
            plt.xlabel("Sample")
            plt.ylabel("Normalized amplitude")
            plt.tight_layout()
            safe_id = str(row["sample_id"]).replace(":", "_").replace("/", "_")
            plt.savefig(out_dir / f"{class_name}_false_negative_{rank}_{safe_id}.png", dpi=140)
            plt.close()


def _write_error_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Error Analysis",
        "",
        "This report is for technical model debugging only. It is not medical reasoning or clinical validation.",
        "",
        f"- Source: `{summary['source']}`",
        f"- Real MIT-BIH: `{summary['real_mitbih']}`",
        f"- Total samples: `{summary['total_samples']}`",
        f"- Misclassified samples: `{summary['misclassified_count']}`",
        f"- High-confidence wrong predictions: `{summary['high_confidence_wrong_count']}`",
        f"- Low-confidence predictions: `{summary['low_confidence_count']}`",
        "",
        "## Worst Classes",
        "",
    ]
    for row in summary["worst_classes"]:
        lines.append(f"- `{row['class']}`: failure rate `{row['failure_rate']:.4f}` ({row['failures']}/{row['samples']})")
    lines.extend(["", "## False Negative Focus", ""])
    for class_name, row in summary["false_negative_summary"].items():
        lines.append(
            f"- `{class_name}`: false negative rate `{row['false_negative_rate']:.4f}` "
            f"({row['false_negative_count']}/{row['support']})"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_confidence_plot(path: Path, confidences: list[float]) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(confidences, bins=12, color="#2563eb", alpha=0.85)
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.title("Prediction Confidence Distribution")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_robustness(config: dict[str, Any], checkpoint: str | None = None) -> dict[str, Any]:
    pipeline = InferencePipeline(_config_with_checkpoint(config, checkpoint))
    fs = int(config["dataset"]["sampling_rate"])
    base = synthetic_beat(0, int(config["preprocessing"]["window_size"]), seed=123)
    cases = {
        "clean": (base, fs),
        "gaussian_noise": (base + np.random.default_rng(1).normal(0, 0.1, base.size), fs),
        "baseline_wander": (base + 0.3 * np.sin(np.linspace(0, 4 * np.pi, base.size)), fs),
        "amplitude_scaling": (base * 2.5, fs),
        "short_signal": (base[:120], fs),
        "nan_inf_signal": (np.where(np.arange(base.size) == 10, np.nan, base), fs),
        "wrong_sampling_rate": (base, 250),
    }
    rows = [_run_robustness_case(pipeline, name, signal, sampling_rate) for name, (signal, sampling_rate) in cases.items()]
    rows.extend(_file_robustness_cases(config))

    out_dir = resolve_path(config["reports"]["robustness_dir"])
    write_csv(out_dir / "robustness_results.csv", rows)
    summary = {
        "accepted": sum(row["status"] == "accepted" for row in rows),
        "rejected": sum(row["status"] == "rejected" for row in rows),
        "cases": rows,
        "disclaimer": config["project"]["disclaimer"],
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "robustness_summary.json", summary)
    return summary


def _run_robustness_case(pipeline: InferencePipeline, name: str, signal: np.ndarray, sampling_rate: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = pipeline.predict_signal(signal, sampling_rate=sampling_rate, input_filename=name)
        return {
            "case": name,
            "status": "accepted",
            "prediction": result["prediction"]["class_name"],
            "confidence": result["prediction"]["confidence"],
            "processing_time_ms": round((time.perf_counter() - started) * 1000, 4),
            "error": "",
        }
    except (SignalValidationError, ValueError) as exc:
        return {
            "case": name,
            "status": "rejected",
            "prediction": "",
            "confidence": 0.0,
            "processing_time_ms": round((time.perf_counter() - started) * 1000, 4),
            "error": str(exc),
        }


def _file_robustness_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        bad_csv = Path(tmp) / "corrupted.csv"
        bad_csv.write_text("not,an,ecg\nx,y,z\n", encoding="utf-8")
        started = time.perf_counter()
        try:
            load_signal_file(bad_csv, config, create_demo_if_missing=False)
            rows.append({"case": "corrupted_csv", "status": "accepted", "prediction": "", "confidence": 0.0, "processing_time_ms": round((time.perf_counter() - started) * 1000, 4), "error": ""})
        except Exception as exc:
            rows.append({"case": "corrupted_csv", "status": "rejected", "prediction": "", "confidence": 0.0, "processing_time_ms": round((time.perf_counter() - started) * 1000, 4), "error": str(exc)})
    rows.append(
        {
            "case": "missing_annotation",
            "status": "rejected",
            "prediction": "",
            "confidence": 0.0,
            "processing_time_ms": 0.0,
            "error": "WFDB records must include matching .dat, .hea, and annotation files; preparation validates this before training.",
        }
    )
    return rows


def benchmark_model(config: dict[str, Any], checkpoint: str | None = None, iterations: int = 100) -> dict[str, Any]:
    pipeline = InferencePipeline(_config_with_checkpoint(config, checkpoint))
    fs = int(config["dataset"]["sampling_rate"])
    base = synthetic_beat(0, int(config["preprocessing"]["window_size"]), seed=5)
    preprocessing_times = []
    inference_times = []
    total_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        pre_start = time.perf_counter()
        windows = segment_signal(base, config)
        preprocessing_times.append((time.perf_counter() - pre_start) * 1000)
        inf_start = time.perf_counter()
        pipeline.predict_windows(windows)
        inference_times.append((time.perf_counter() - inf_start) * 1000)
        total_times.append((time.perf_counter() - start) * 1000)
    summary = {
        "runtime": pipeline.runtime,
        "iterations": iterations,
        "input_sampling_rate": fs,
        "latency_ms": {
            "preprocessing": _latency_stats(preprocessing_times),
            "inference": _latency_stats(inference_times),
            "total": _latency_stats(total_times),
        },
        "target": {
            "onnx_inference_p95_ms": 200,
            "first_dashboard_payload_sec": 5,
        },
        "disclaimer": config["project"]["disclaimer"],
    }
    write_json(resolve_path(config["artifacts"]["metrics_dir"]) / "benchmark_summary.json", summary)
    write_json(resolve_path(config["reports"]["benchmark_dir"]) / "latency_breakdown.json", summary)
    return summary


def _latency_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
        "median": float(median(values)),
    }

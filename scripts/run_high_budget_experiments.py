"""Run higher-budget MIT-BIH model-quality experiments.

This runner is intentionally separate from the production training command. It
keeps candidate checkpoints isolated, applies an explicit promotion rule, and
preserves artifacts/models/best_model.pt.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.config_loader import load_config, resolve_path
from src.evaluation import evaluate_model, tune_thresholds
from src.training import train_model
from src.utils.io import write_json


BASELINE_MACRO_F1 = 0.2824
BASELINE_METRICS = {
    "accuracy": 0.6284,
    "macro_f1": 0.2824,
    "weighted_f1": 0.6057,
    "per_class_f1": {
        "N": 0.8975,
        "V": 0.4527,
        "A": 0.0014,
        "L": 0.0,
        "R": 0.0606,
    },
}


def _display_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = resolve_path(path).resolve()
    try:
        return str(resolved.relative_to(resolve_path(".").resolve()))
    except ValueError:
        return str(resolved)


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "resnet1d_maxabs_focal_balanced": {
        "model_type": "resnet1d",
        "dropout": 0.35,
        "normalization": "maxabs",
        "learning_rate": 7e-4,
        "loss": "focal",
        "focal_gamma": 1.5,
        "sampler_weight_power": 1.0,
    },
    "resnet1d_zscore_focal_balanced": {
        "model_type": "resnet1d",
        "dropout": 0.35,
        "normalization": "zscore",
        "learning_rate": 7e-4,
        "loss": "focal",
        "focal_gamma": 1.5,
        "sampler_weight_power": 1.0,
    },
    "inceptiontime_maxabs_focal_balanced": {
        "model_type": "inceptiontime",
        "dropout": 0.25,
        "normalization": "maxabs",
        "learning_rate": 8e-4,
        "loss": "focal",
        "focal_gamma": 1.5,
        "sampler_weight_power": 1.0,
    },
    "cnn_lstm_maxabs_focal_balanced": {
        "model_type": "cnn_lstm",
        "dropout": 0.35,
        "normalization": "maxabs",
        "learning_rate": 7e-4,
        "loss": "focal",
        "focal_gamma": 1.5,
        "sampler_weight_power": 1.0,
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dir(path: str | Path) -> Path:
    target = resolve_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _configure_experiment(
    base_config: dict[str, Any],
    name: str,
    spec: dict[str, Any],
    epochs: int,
    cap: int | None,
    budget_label: str,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    candidate_dir = _ensure_dir("artifacts/models/candidates")
    experiment_dir = _ensure_dir(f"reports/experiments/high_budget/{name}")
    metrics_dir = _ensure_dir(f"artifacts/metrics/high_budget/{name}")
    threshold_path = resolve_path(f"artifacts/evaluation/high_budget/{name}_thresholds.json")
    threshold_path.parent.mkdir(parents=True, exist_ok=True)

    model_cfg = config.setdefault("model", {})
    model_cfg["type"] = spec["model_type"]
    model_cfg["dropout"] = float(spec["dropout"])
    model_cfg["checkpoint"] = str(candidate_dir / f"{name}.pt")
    model_cfg["threshold_path"] = str(threshold_path)

    preprocessing_cfg = config.setdefault("preprocessing", {})
    preprocessing_cfg["normalization"] = spec["normalization"]

    training_cfg = config.setdefault("training", {})
    training_cfg["epochs"] = int(epochs)
    training_cfg["learning_rate"] = float(spec["learning_rate"])
    training_cfg["loss"] = spec["loss"]
    training_cfg["focal_gamma"] = float(spec["focal_gamma"])
    training_cfg["weighted_sampler"] = True
    training_cfg["sampler_weight_power"] = float(spec["sampler_weight_power"])
    training_cfg["early_stopping_monitor"] = "macro_f1"
    training_cfg["scheduler"] = "plateau"
    training_cfg["gradient_clip_norm"] = 1.0
    training_cfg["max_class_weight"] = 12.0
    training_cfg["batch_size"] = min(int(training_cfg.get("batch_size", 128)), 128)
    training_cfg["eval_batch_size"] = int(training_cfg.get("eval_batch_size", 512))
    training_cfg["max_train_samples_per_class"] = None if cap is None else int(cap)

    augmentation_cfg = training_cfg.setdefault("augmentation", {})
    augmentation_cfg["enabled"] = True
    augmentation_cfg["gaussian_noise_std"] = 0.008
    augmentation_cfg["amplitude_scale_min"] = 0.97
    augmentation_cfg["amplitude_scale_max"] = 1.03
    augmentation_cfg["baseline_wander_amplitude"] = 0.0
    augmentation_cfg["time_shift_max"] = 4

    inference_cfg = config.setdefault("inference", {})
    inference_cfg["batch_size"] = min(int(inference_cfg.get("batch_size", 512)), 512)

    reports_cfg = config.setdefault("reports", {})
    reports_cfg["evaluation_dir"] = str(experiment_dir / "evaluation")
    reports_cfg["error_analysis_dir"] = str(experiment_dir / "error_analysis")
    reports_cfg["explainability_dir"] = str(experiment_dir / "explainability")

    artifacts_cfg = config.setdefault("artifacts", {})
    artifacts_cfg["metrics_dir"] = str(metrics_dir)
    artifacts_cfg["evaluation_dir"] = str(experiment_dir / "artifacts_evaluation")

    config["research"] = {
        "experiment_name": name,
        "budget_label": budget_label,
        "baseline_macro_f1": BASELINE_MACRO_F1,
        "promotion_rule": {
            "macro_f1_gt": BASELINE_MACRO_F1,
            "required_nonzero_f1": ["A", "L", "R", "V"],
            "anti_collapse_required": True,
            "min_unique_predicted_classes": 4,
        },
    }
    return config


def _per_class_f1(metrics: dict[str, Any]) -> dict[str, float]:
    report = metrics.get("classification_report", {})
    return {
        label: float(report.get(label, {}).get("f1-score", 0.0))
        for label in ["N", "V", "A", "L", "R"]
    }


def _promotion_status(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    macro_f1 = float(metrics.get("macro_f1", 0.0))
    per_class = _per_class_f1(metrics)
    anti = metrics.get("anti_collapse_check", {}) or {}
    unique_predicted = int(anti.get("unique_classes_predicted", 0))

    if macro_f1 <= BASELINE_MACRO_F1:
        reasons.append(f"macro_f1 {macro_f1:.4f} <= baseline {BASELINE_MACRO_F1:.4f}")
    for label in ["A", "L", "R", "V"]:
        if per_class.get(label, 0.0) <= 0.0:
            reasons.append(f"{label} F1 is zero")
    if anti.get("status") != "passed":
        reasons.append(f"anti-collapse status is {anti.get('status', 'missing')}")
    if unique_predicted < 4:
        reasons.append(f"only {unique_predicted} predicted classes")
    return not reasons, reasons


def _row_from_metrics(name: str, config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    per_class = _per_class_f1(metrics)
    promotion_passed, promotion_reasons = _promotion_status(metrics)
    anti = metrics.get("anti_collapse_check", {}) or {}
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    preprocessing_cfg = config.get("preprocessing", {})

    row: dict[str, Any] = {
        "experiment": name,
        "status": "completed",
        "model_type": model_cfg.get("type"),
        "normalization": preprocessing_cfg.get("normalization"),
        "loss": training_cfg.get("loss"),
        "epochs": training_cfg.get("epochs"),
        "max_train_samples_per_class": training_cfg.get("max_train_samples_per_class"),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "macro_f1": float(metrics.get("macro_f1", 0.0)),
        "weighted_f1": float(metrics.get("weighted_f1", 0.0)),
        "roc_auc_ovr_macro": float(metrics.get("roc_auc_ovr_macro", 0.0)),
        "N_f1": per_class.get("N", 0.0),
        "V_f1": per_class.get("V", 0.0),
        "A_f1": per_class.get("A", 0.0),
        "L_f1": per_class.get("L", 0.0),
        "R_f1": per_class.get("R", 0.0),
        "unique_predicted_classes": anti.get("unique_classes_predicted"),
        "anti_collapse_status": anti.get("status"),
        "promotion_passed": promotion_passed,
        "promotion_reasons": promotion_reasons,
        "checkpoint": _display_path(model_cfg.get("checkpoint")),
        "evaluation_dir": _display_path(config.get("reports", {}).get("evaluation_dir")),
        "generated_at": _utc_now(),
    }
    threshold_summary = metrics.get("threshold_summary") or {}
    if threshold_summary:
        row["threshold_val_macro_f1"] = threshold_summary.get("best_macro_f1")
        row["threshold_argmax_val_macro_f1"] = threshold_summary.get("argmax_macro_f1")
    return row


def _write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            safe_row = {
                key: json.dumps(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(safe_row)


def _write_markdown(rows: list[dict[str, Any]], summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# High-Budget MIT-BIH Research Experiments",
        "",
        "These experiments are grouped-record MIT-BIH research runs. They do not change the preserved production baseline checkpoint unless the explicit promotion rule passes.",
        "",
        f"- Generated: {summary['generated_at']}",
        f"- Budget label: {summary['budget_label']}",
        f"- Baseline macro F1: {BASELINE_MACRO_F1:.4f}",
        f"- Promoted: {summary['promoted']}",
        "",
        "| Experiment | Status | Macro F1 | Accuracy | N F1 | V F1 | A F1 | L F1 | R F1 | Promotion |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if row.get("status") != "completed":
            lines.append(
                f"| {row.get('experiment')} | {row.get('status')} |  |  |  |  |  |  |  | no |"
            )
            continue
        lines.append(
            "| {experiment} | {status} | {macro_f1:.4f} | {accuracy:.4f} | {N_f1:.4f} | {V_f1:.4f} | {A_f1:.4f} | {L_f1:.4f} | {R_f1:.4f} | {promotion} |".format(
                experiment=row.get("experiment"),
                status=row.get("status"),
                macro_f1=float(row.get("macro_f1", 0.0)),
                accuracy=float(row.get("accuracy", 0.0)),
                N_f1=float(row.get("N_f1", 0.0)),
                V_f1=float(row.get("V_f1", 0.0)),
                A_f1=float(row.get("A_f1", 0.0)),
                L_f1=float(row.get("L_f1", 0.0)),
                R_f1=float(row.get("R_f1", 0.0)),
                promotion="yes" if row.get("promotion_passed") else "no",
            )
        )
    if summary.get("best_candidate"):
        lines.extend(["", f"Best candidate: `{summary['best_candidate']['experiment']}`."])
    if summary.get("promotion_notes"):
        lines.extend(["", "## Promotion Notes", ""])
        for note in summary["promotion_notes"]:
            lines.append(f"- {note}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_single_experiment(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{args.experiment}'. Options: {sorted(EXPERIMENTS)}")
    budget_label = args.budget_label or (
        "full_record_split" if args.cap is None else f"larger_cpu_budget_epochs_{args.epochs}_cap_{args.cap}"
    )
    exp_config = _configure_experiment(
        config,
        args.experiment,
        EXPERIMENTS[args.experiment],
        epochs=args.epochs,
        cap=args.cap,
        budget_label=budget_label,
    )

    if args.evaluate_existing:
        checkpoint_path = resolve_path(exp_config["model"]["checkpoint"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot evaluate existing checkpoint because it does not exist: {checkpoint_path}")
        print(f"[high-budget] evaluating existing checkpoint for {args.experiment}", flush=True)
    else:
        print(f"[high-budget] training {args.experiment}", flush=True)
        train_model(exp_config)
    checkpoint = exp_config["model"]["checkpoint"]
    print(f"[high-budget] tuning thresholds {args.experiment}", flush=True)
    tune_thresholds(exp_config, checkpoint=checkpoint)
    print(f"[high-budget] evaluating {args.experiment}", flush=True)
    metrics = evaluate_model(exp_config, checkpoint=checkpoint)
    row = _row_from_metrics(args.experiment, exp_config, metrics)
    if args.row_output:
        write_json(args.row_output, row)
    return 0


def _run_all(args: argparse.Namespace) -> int:
    root = _ensure_dir("reports/experiments/high_budget")
    selected = args.experiments or list(EXPERIMENTS)
    budget_label = args.budget_label or (
        "full_record_split" if args.cap is None else f"larger_cpu_budget_epochs_{args.epochs}_cap_{args.cap}"
    )

    rows: list[dict[str, Any]] = []
    for name in selected:
        row_path = root / f"{name}.row.json"
        cmd = [
            sys.executable,
            "-m",
            "scripts.run_high_budget_experiments",
            "--config",
            args.config,
            "--single-experiment",
            name,
            "--epochs",
            str(args.epochs),
            "--row-output",
            str(row_path),
            "--budget-label",
            budget_label,
        ]
        if args.cap is not None:
            cmd.extend(["--cap", str(args.cap)])
        print(f"[high-budget] starting {name}", flush=True)
        result = subprocess.run(cmd, cwd=resolve_path("."))
        if row_path.exists():
            rows.append(json.loads(row_path.read_text(encoding="utf-8")))
        else:
            rows.append(
                {
                    "experiment": name,
                    "status": "failed",
                    "exit_code": result.returncode,
                    "promotion_passed": False,
                    "promotion_reasons": [f"subprocess failed with exit code {result.returncode}"],
                    "generated_at": _utc_now(),
                }
            )

    completed = [row for row in rows if row.get("status") == "completed"]
    passing = [row for row in completed if row.get("promotion_passed")]
    best = None
    promoted = False
    promotion_notes: list[str] = []
    if passing:
        best = max(passing, key=lambda row: float(row.get("macro_f1", 0.0)))
        destination = resolve_path("artifacts/models/best_model_research.pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolve_path(str(best["checkpoint"])), destination)
        promoted = True
        promotion_notes.append(
            f"Copied {best['checkpoint']} to artifacts/models/best_model_research.pt after passing the promotion rule."
        )
    else:
        if completed:
            best = max(completed, key=lambda row: float(row.get("macro_f1", 0.0)))
        promotion_notes.append("No candidate passed the promotion rule; artifacts/models/best_model.pt was preserved.")
        promotion_notes.append("artifacts/models/best_model_research.pt was not updated by this run.")

    summary = {
        "generated_at": _utc_now(),
        "budget_label": budget_label,
        "baseline_metrics": BASELINE_METRICS,
        "experiments": rows,
        "best_candidate": best,
        "promoted": promoted,
        "promotion_notes": promotion_notes,
        "preserved_checkpoint": "artifacts/models/best_model.pt",
    }
    write_json(root / "experiment_summary.json", summary)
    _write_rows_csv(rows, root / "experiment_results.csv")
    _write_markdown(rows, summary, root / "experiment_summary.md")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--cap", type=int, default=4000, help="Per-class training cap. Use --cap 0 for full data.")
    parser.add_argument("--experiments", nargs="*", choices=sorted(EXPERIMENTS))
    parser.add_argument("--budget-label")
    parser.add_argument("--single-experiment", dest="experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--row-output")
    parser.add_argument("--evaluate-existing", action="store_true")
    args = parser.parse_args()
    if args.cap == 0:
        args.cap = None
    return args


def main() -> int:
    args = parse_args()
    if args.experiment:
        return _run_single_experiment(args)
    return _run_all(args)


if __name__ == "__main__":
    raise SystemExit(main())

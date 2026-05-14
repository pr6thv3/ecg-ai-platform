from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import pandas as pd

from src.config import ensure_config_dirs, load_config, resolve_path
from src.data.dataset import leakage_report, load_dataset, mitbih_inventory, split_dataset
from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ECG dataset availability, shape, labels, and leakage.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_config_dirs(config)
    bundle = load_dataset(config)
    splits = split_dataset(bundle, config)
    class_names = config["model"]["class_names"]
    class_distribution = _named_distribution(bundle.y, class_names)
    warnings = list(bundle.warnings)
    for class_name, count in class_distribution["counts"].items():
        if count == 0:
            warnings.append(f"Dataset has zero samples for class {class_name}.")
        elif count < 50:
            warnings.append(f"Dataset has only {count} samples for class {class_name}; metrics may be unstable.")
    inventory = mitbih_inventory(
        config["dataset"]["mitbih_path"],
        [str(record) for record in config["dataset"].get("records", [])],
        str(config["dataset"].get("annotation_extension", "atr")),
    )
    report = {
        "status": "passed",
        "source": bundle.source,
        "real_mitbih": bundle.source == "mitbih",
        "sample_count": int(len(bundle.y)),
        "record_count": int(len(set(bundle.record_ids.tolist()))),
        "sampling_rate": int(config["dataset"]["sampling_rate"]),
        "split_strategy": config["dataset"]["split"].get("group_by", "random"),
        "window_shape": list(bundle.X.shape),
        "class_distribution": class_distribution,
        "mitbih_inventory": inventory,
        "warnings": warnings,
        "leakage": leakage_report(splits),
        "disclaimer": config["project"]["disclaimer"],
    }
    if bundle.source != "mitbih":
        report["status"] = "passed_with_synthetic_fallback"
    write_json("reports/evaluation/dataset_validation.json", report)
    evaluation_dir = resolve_path(config["artifacts"].get("evaluation_dir", "artifacts/evaluation"))
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    write_json(evaluation_dir / "dataset_summary.json", report)
    write_json(evaluation_dir / "splits.json", report["leakage"])
    _save_class_distribution(evaluation_dir, class_distribution)
    write_json(config["artifacts"]["split_manifest"], report["leakage"])
    print(json.dumps(report, indent=2))


def _named_distribution(y, class_names: list[str]) -> dict:
    total = int(len(y))
    counts = {name: int((y == idx).sum()) for idx, name in enumerate(class_names)}
    percentages = {name: (round(count / total * 100.0, 4) if total else 0.0) for name, count in counts.items()}
    return {"total": total, "counts": counts, "percentages": percentages}


def _save_class_distribution(out_dir, distribution: dict) -> None:
    rows = [
        {"class": class_name, "count": count, "percentage": distribution["percentages"][class_name]}
        for class_name, count in distribution["counts"].items()
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "dataset_class_distribution.csv", index=False)
    plt.figure(figsize=(7, 4))
    plt.bar(frame["class"], frame["count"], color="#2563eb")
    plt.xlabel("Class")
    plt.ylabel("Beat/window count")
    plt.title("Dataset Class Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "dataset_class_distribution.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()

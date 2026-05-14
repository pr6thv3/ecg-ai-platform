"""Validate MIT-BIH WFDB files and write split/leakage preparation artifacts."""

from __future__ import annotations

import argparse
import json
import sys

from src.config import ensure_config_dirs, load_config, resolve_path
from src.data.dataset import leakage_report, load_dataset, mitbih_inventory, split_dataset
from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MIT-BIH dataset: validate WFDB files, split, and check leakage.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when no complete MIT-BIH records are available.")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_config_dirs(config)
    dataset_cfg = config["dataset"]
    inventory = mitbih_inventory(
        resolve_path(dataset_cfg["mitbih_path"]),
        [str(record) for record in dataset_cfg.get("records", [])],
        str(dataset_cfg.get("annotation_extension", "atr")),
    )

    report = {
        "status": "ready" if inventory["ready"] else "missing_required_files",
        "source": "mitbih" if inventory["ready"] else "unavailable",
        "real_mitbih": bool(inventory["ready"]),
        "inventory": inventory,
        "required_layout": "Place each MIT-BIH record as matching <record>.dat, <record>.hea, and <record>.atr under datasets/mit-bih/.",
        "disclaimer": config["project"]["disclaimer"],
    }

    if inventory["ready"]:
        bundle = load_dataset(config)
        splits = split_dataset(bundle, config)
        leakage = leakage_report(splits)
        report.update(
            {
                "status": "ready" if leakage["status"] == "passed" else "leakage_failed",
                "loaded_samples": int(len(bundle.y)),
                "record_count": int(len(set(bundle.record_ids.tolist()))),
                "sampling_rate": int(config["dataset"]["sampling_rate"]),
                "split_strategy": config["dataset"]["split"].get("group_by", "random"),
                "leakage": leakage,
                "warnings": bundle.warnings,
            }
        )
        write_json(config["artifacts"]["split_manifest"], leakage)
        write_json("reports/evaluation/leakage_check.json", leakage)
        write_json(resolve_path(config["artifacts"].get("evaluation_dir", "artifacts/evaluation")) / "splits.json", leakage)
    else:
        if bool(dataset_cfg.get("allow_synthetic_fallback", False)):
            report["warnings"] = [
                "No complete MIT-BIH WFDB record set was found. Synthetic demo fallback is explicitly enabled."
            ]
        else:
            report["warnings"] = [
                "No complete MIT-BIH WFDB record set was found. Place matching .dat, .hea, and .atr files under datasets/mit-bih or explicitly enable synthetic demo fallback."
            ]

    write_json("reports/evaluation/mitbih_preparation.json", report)
    write_json(resolve_path(config["artifacts"].get("evaluation_dir", "artifacts/evaluation")) / "mitbih_preparation.json", report)
    print(json.dumps(report, indent=2))
    if args.strict and not inventory["ready"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

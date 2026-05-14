from __future__ import annotations

import argparse
import json

from src.config import ensure_config_dirs, load_config
from src.data.dataset import leakage_report, load_dataset, split_dataset
from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Write train/val/test record_id leakage report.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_config_dirs(config)
    report = leakage_report(split_dataset(load_dataset(config), config))
    write_json(config["artifacts"]["split_manifest"], report)
    write_json("reports/evaluation/leakage_check.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from src.config import ensure_config_dirs, load_config
from src.inference import InferencePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full ECG smoke path through preprocessing and inference.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_config_dirs(config)
    sample = config["dataset"]["sample_file"]
    result = InferencePipeline(config).predict_file(sample)
    result["smoke_test"] = {
        "status": "passed",
        "sample_file": sample,
        "note": "Uses the configured sample file; synthetic demo generation is only available when explicitly enabled.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

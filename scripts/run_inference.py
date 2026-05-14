from __future__ import annotations

import argparse
import json

from src.config import ensure_config_dirs, load_config
from src.inference import InferencePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ECG inference on a CSV/TXT ECG signal.")
    parser.add_argument("--input", required=True, help="Path to ECG signal file. A demo file is created if missing and fallback is enabled.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_config_dirs(config)
    result = InferencePipeline(config).predict_file(args.input)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

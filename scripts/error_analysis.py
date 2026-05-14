from __future__ import annotations

import argparse
import json

from src.config import ensure_config_dirs, load_config
from src.evaluation import run_error_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ECG error-analysis artifacts.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_config_dirs(config)
    print(json.dumps(run_error_analysis(config, checkpoint=args.checkpoint), indent=2))


if __name__ == "__main__":
    main()

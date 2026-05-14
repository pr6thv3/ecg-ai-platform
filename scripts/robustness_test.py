from __future__ import annotations

import argparse
import json

from src.config import ensure_config_dirs, load_config
from src.evaluation import run_robustness


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ECG robustness checks against noisy/corrupted inputs.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_config_dirs(config)
    print(json.dumps(run_robustness(config, checkpoint=args.checkpoint), indent=2))


if __name__ == "__main__":
    main()

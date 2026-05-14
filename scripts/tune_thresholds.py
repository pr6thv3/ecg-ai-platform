from __future__ import annotations

import argparse
import json

from src.config import ensure_config_dirs, load_config
from src.evaluation import tune_thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune per-class decision thresholds on the validation split.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_config_dirs(config)
    print(json.dumps(tune_thresholds(config, checkpoint=args.checkpoint), indent=2))


if __name__ == "__main__":
    main()

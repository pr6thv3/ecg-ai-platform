from __future__ import annotations

import argparse
import json

from src.config import ensure_config_dirs, load_config
from src.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ECG classifier with reproducible split and leakage guardrails.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_config_dirs(config)
    print(json.dumps(train_model(config), indent=2))


if __name__ == "__main__":
    main()

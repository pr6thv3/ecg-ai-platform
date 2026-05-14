from __future__ import annotations

import argparse
import json

from src.config import ensure_config_dirs, load_config
from src.explainability import explain_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ECG prediction explanation artifacts.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_config_dirs(config)
    print(json.dumps(explain_prediction(config, args.input), indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from src.config import ensure_config_dirs, load_config
from src.evaluation import benchmark_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ECG preprocessing and model inference latency.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_config_dirs(config)
    print(json.dumps(benchmark_model(config, checkpoint=args.checkpoint, iterations=args.iterations), indent=2))


if __name__ == "__main__":
    main()

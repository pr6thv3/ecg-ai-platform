from __future__ import annotations

import argparse
import json

from src.config import ensure_config_dirs, load_config
from src.data.dataset import leakage_report, load_dataset, split_dataset
from src.evaluation import benchmark_model, evaluate_model
from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MIT-BIH readiness artifacts: evaluation, leakage, benchmark.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_config_dirs(config)
    bundle = load_dataset(config)
    splits = split_dataset(bundle, config)
    leakage = leakage_report(splits)
    write_json(config["artifacts"]["split_manifest"], leakage)
    write_json("reports/evaluation/leakage_check.json", leakage)
    evaluation = evaluate_model(config, checkpoint=args.checkpoint)
    benchmark = benchmark_model(config, checkpoint=args.checkpoint, iterations=args.iterations)
    summary = {
        "status": "completed",
        "source": bundle.source,
        "real_mitbih": bundle.source == "mitbih",
        "evaluation_artifact": "reports/evaluation/mitbih_evaluation_summary.json",
        "leakage_artifact": "reports/evaluation/leakage_check.json",
        "benchmark_artifact": "artifacts/metrics/benchmark_summary.json",
        "warning": "Synthetic fallback was used; provide MIT-BIH WFDB records before citing model performance."
        if bundle.source != "mitbih"
        else "",
        "macro_f1": evaluation.get("macro_f1"),
        "onnx_inference_p95_ms": benchmark["latency_ms"]["inference"]["p95"],
    }
    write_json("reports/evaluation/mitbih_artifact_generation_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

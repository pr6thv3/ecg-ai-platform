import argparse
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.model_manager import ModelManager


def run_many(label: str, fn, sample: np.ndarray, runs: int) -> dict:
    for _ in range(10):
        fn(sample)

    latencies = []
    start = time.perf_counter()
    for _ in range(runs):
        item_start = time.perf_counter()
        fn(sample)
        latencies.append((time.perf_counter() - item_start) * 1000)
    total = time.perf_counter() - start

    return {
        "runtime": label,
        "mean_ms": statistics.fmean(latencies),
        "p50_ms": statistics.median(latencies),
        "p95_ms": sorted(latencies)[int(runs * 0.95) - 1],
        "p99_ms": sorted(latencies)[int(runs * 0.99) - 1],
        "throughput": runs / total,
    }


def benchmark(model_path: str, runs: int) -> None:
    model_mgr = ModelManager()
    model_mgr.load_model(model_path, device="cpu")

    sample = np.zeros((360,), dtype=np.float32)
    results = []

    if model_mgr.onnx_session is not None:
        results.append(run_many("ONNX Runtime CPU", model_mgr.onnx_predict, sample, runs))
    if model_mgr.model is not None:
        results.append(run_many("PyTorch CPU", model_mgr.predict, sample, runs))

    if not results:
        raise RuntimeError("No runnable model runtime was loaded.")

    print("| Runtime | Mean ms | P50 ms | P95 ms | P99 ms | Inferences/sec |")
    print("| :--- | ---: | ---: | ---: | ---: | ---: |")
    for row in results:
        print(
            f"| {row['runtime']} | {row['mean_ms']:.4f} | {row['p50_ms']:.4f} | "
            f"{row['p95_ms']:.4f} | {row['p99_ms']:.4f} | {row['throughput']:.2f} |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ECG model inference latency.")
    parser.add_argument("--model-path", default=os.path.join("models", "ecg_cnn.onnx"))
    parser.add_argument("--runs", type=int, default=1000)
    args = parser.parse_args()
    benchmark(args.model_path, args.runs)

# Performance Benchmarks

This document details the performance characteristics of the ECG AI Platform, focusing on inference latency, system responsiveness, and model optimization metrics.

## 1. Inference latency

| Runtime | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Beats/sec |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PyTorch | [RUN: pytest scripts/benchmark_inference.py] | [RUN: pytest scripts/benchmark_inference.py] | [RUN: pytest scripts/benchmark_inference.py] | [RUN: pytest scripts/benchmark_inference.py] | [RUN: pytest scripts/benchmark_inference.py] |
| ONNX | [RUN: pytest scripts/benchmark_inference.py] | [RUN: pytest scripts/benchmark_inference.py] | [RUN: pytest scripts/benchmark_inference.py] | [RUN: pytest scripts/benchmark_inference.py] | [RUN: pytest scripts/benchmark_inference.py] |

ONNX Runtime reduces mean inference latency by [RUN: pytest scripts/benchmark_inference.py]%, enabling real-time classification at [RUN: pytest scripts/benchmark_inference.py] beats/sec on CPU.

## 2. System performance

| Metric | Value |
| :--- | :--- |
| End-to-end latency (sample→output) | [RUN: pytest scripts/benchmark_system.py] ms |
| WebSocket update interval | [RUN: pytest scripts/benchmark_system.py] ms |
| Backend cold start time | [RUN: pytest scripts/benchmark_system.py] sec |
| PDF report generation | [RUN: pytest scripts/benchmark_system.py] sec |
| ONNX model file size | [RUN: pytest scripts/benchmark_system.py] MB |
| PyTorch model file size | [RUN: pytest scripts/benchmark_system.py] MB |

## 3. Test coverage

| Module | Coverage |
| :--- | :--- |
| backend/preprocessing/ | [RUN: pytest --cov=backend/preprocessing] |
| backend/inference/ | [RUN: pytest --cov=backend/inference] |
| backend/api/ | [RUN: pytest --cov=backend/api] |
| frontend/components/ | [RUN: npm run test:coverage] |
| Overall | [RUN: pytest --cov] |

## 4. Ablation study results

| Pipeline variant | Accuracy | Macro-F1 | Latency (ms/beat) |
| :--- | :--- | :--- | :--- |
| No filtering | [RUN: pytest scripts/ablation_study.py]% | [RUN: pytest scripts/ablation_study.py] | [RUN: pytest scripts/ablation_study.py] |
| No segmentation | [RUN: pytest scripts/ablation_study.py]% | [RUN: pytest scripts/ablation_study.py] | [RUN: pytest scripts/ablation_study.py] |
| Full pipeline | [RUN: pytest scripts/ablation_study.py]% | [RUN: pytest scripts/ablation_study.py] | [RUN: pytest scripts/ablation_study.py] |

Filtering removes baseline wander and high-frequency noise improving model accuracy, while segmentation enables the 1D CNN to focus purely on morphological features.

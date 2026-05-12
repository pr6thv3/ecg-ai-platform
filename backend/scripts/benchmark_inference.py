import time
import torch
import numpy as np
from tabulate import tabulate
import os
import sys

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.model_manager import ModelManager, USE_ONNX

def benchmark():
    print("Starting Inference Benchmark...")
    
    # Initialize ModelManager
    model_mgr = ModelManager()
    
    # Load PyTorch model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ecg_cnn.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please ensure the model is trained and saved.")
        return

    model_mgr.load_model(model_path, device='cpu')
    
    # Load ONNX model
    onnx_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ecg_cnn.onnx')
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}. Please run export_onnx.py first.")
        return
        
    # Model manager initializes ONNX session automatically if file exists and ONNX is installed.
    # However, let's explicitly initialize it here if it wasn't done during load_model.
    if model_mgr.onnx_session is None:
        try:
            import onnxruntime as ort
            model_mgr.onnx_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            print(f"Loaded ONNX model from {onnx_path}")
        except Exception as e:
            print(f"Failed to load ONNX model: {e}")
            return
            
    num_runs = 1000
    dummy_input = np.zeros((360,), dtype=np.float32)
    
    print(f"Running {num_runs} inferences for PyTorch...")
    # Warmup
    for _ in range(10):
        model_mgr.predict(dummy_input)
        
    start_time = time.time()
    for _ in range(num_runs):
        model_mgr.predict(dummy_input)
    pt_total_time = time.time() - start_time
    pt_latency = (pt_total_time / num_runs) * 1000
    pt_throughput = num_runs / pt_total_time
    
    print(f"Running {num_runs} inferences for ONNX...")
    # Warmup
    for _ in range(10):
        model_mgr.onnx_predict(dummy_input)
        
    start_time = time.time()
    for _ in range(num_runs):
        model_mgr.onnx_predict(dummy_input)
    onnx_total_time = time.time() - start_time
    onnx_latency = (onnx_total_time / num_runs) * 1000
    onnx_throughput = num_runs / onnx_total_time
    
    print("\n--- Benchmark Results ---")
    table = [
        ["PyTorch (CPU)", f"{pt_latency:.4f}", f"{pt_throughput:.2f}"],
        ["ONNX Runtime (CPU)", f"{onnx_latency:.4f}", f"{onnx_throughput:.2f}"]
    ]
    headers = ["Runtime", "Avg Latency (ms/inf)", "Throughput (inf/sec)"]
    print(tabulate(table, headers, tablefmt="grid"))

if __name__ == "__main__":
    benchmark()

import os
import time
import json
import psutil
import torch
import numpy as np
import sys

# Add backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.model_manager import ModelManager

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # MB

def run_system_benchmark():
    print("Starting System Benchmark...")
    
    results = {
        "model_size_mb": {},
        "latency_ms": {},
        "peak_memory_mb": {}
    }
    
    pt_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ecg_cnn.pth')
    onnx_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'ecg_cnn.onnx')
    
    if os.path.exists(pt_path):
        results["model_size_mb"]["pytorch"] = os.path.getsize(pt_path) / (1024 * 1024)
    if os.path.exists(onnx_path):
        results["model_size_mb"]["onnx"] = os.path.getsize(onnx_path) / (1024 * 1024)
        
    model_mgr = ModelManager()
    model_mgr.load_model(pt_path, device='cpu')
    
    if model_mgr.onnx_session is None and os.path.exists(onnx_path):
        import onnxruntime as ort
        model_mgr.onnx_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
    dummy_input = np.zeros((360,), dtype=np.float32)
    
    # Benchmark PyTorch
    mem_before = get_process_memory()
    start_time = time.time()
    for _ in range(500):
        model_mgr.predict(dummy_input)
    end_time = time.time()
    mem_after = get_process_memory()
    
    results["latency_ms"]["pytorch"] = ((end_time - start_time) / 500) * 1000
    results["peak_memory_mb"]["pytorch_process"] = max(mem_before, mem_after)
    
    # Benchmark ONNX
    mem_before = get_process_memory()
    start_time = time.time()
    for _ in range(500):
        model_mgr.onnx_predict(dummy_input)
    end_time = time.time()
    mem_after = get_process_memory()
    
    results["latency_ms"]["onnx"] = ((end_time - start_time) / 500) * 1000
    results["peak_memory_mb"]["onnx_process"] = max(mem_before, mem_after)
    
    save_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'system_benchmark.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"System benchmark complete. Results saved to {save_path}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_system_benchmark()

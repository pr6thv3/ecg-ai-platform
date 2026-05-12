import os
import argparse
import logging
import torch
import numpy as np
import onnx
import onnxruntime as ort

from models.train import ECGNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def export_to_onnx(model_path: str, output_path: str):
    logger.info(f"Loading PyTorch model from {model_path}...")
    device = torch.device("cpu")
    
    model = ECGNet(num_classes=5)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        logger.warning(f"Model file {model_path} not found. Using randomly initialized weights for testing.")
        
    model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 360, device=device)
    
    logger.info(f"Exporting model to ONNX format at {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["ecg_beat"],
        output_names=["class_probs"],
        dynamic_axes={
            "ecg_beat": {0: "batch_size"},
            "class_probs": {0: "batch_size"}
        }
    )
    logger.info("Export completed.")
    
    # Validation
    logger.info("Validating ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model is valid.")
    
    # Print stats
    model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    num_nodes = len(onnx_model.graph.node)
    logger.info(f"ONNX Model Size: {model_size_mb:.2f} MB")
    logger.info(f"Number of nodes in ONNX graph: {num_nodes}")
    
    # Comparison
    logger.info("Comparing PyTorch and ONNX Runtime outputs...")
    with torch.no_grad():
        torch_output = model(dummy_input).numpy()
        
    ort_session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"ecg_beat": dummy_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    max_diff = np.max(np.abs(torch_output - ort_output))
    logger.info(f"Max absolute difference between PyTorch and ONNX: {max_diff}")
    
    assert max_diff < 1e-5, f"Difference too large: {max_diff}"
    logger.info("Validation successful: Outputs match within tolerance.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch ECGNet to ONNX")
    parser.add_argument("--model-path", type=str, default="models/ecg_model.pth", help="Path to input .pth model")
    parser.add_argument("--output-path", type=str, default="models/ecg_model.onnx", help="Path to output .onnx model")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    export_to_onnx(args.model_path, args.output_path)

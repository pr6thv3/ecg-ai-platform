import torch

from src.config import load_config
from src.models import build_model


def test_supported_models_forward_pass():
    config = load_config("configs/default.yaml")
    x = torch.randn(2, config["model"]["input_size"])
    for model_type in ("baseline_cnn", "resnet1d", "inceptiontime", "cnn_lstm"):
        model = build_model(model_type, num_classes=len(config["model"]["class_names"]), input_size=config["model"]["input_size"])
        logits = model(x)
        assert logits.shape == (2, len(config["model"]["class_names"]))

import pytest
import torch

from src.config import load_config
from src.inference import InferencePipeline
from src.models import build_model


def test_checkpoint_metadata_mismatch_fails(tmp_path):
    config = load_config("configs/default.yaml")
    model = build_model("baseline_cnn", num_classes=len(config["model"]["class_names"]), input_size=config["model"]["input_size"])
    checkpoint = tmp_path / "bad.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "baseline_cnn",
            "class_order": ["X", "V", "A", "L", "R"],
            "class_mapping": config["dataset"]["class_mapping"],
            "input_size": config["model"]["input_size"],
            "preprocessing": {
                "window_size": config["preprocessing"]["window_size"],
                "sampling_rate": config["dataset"]["sampling_rate"],
            },
        },
        checkpoint,
    )
    config["model"]["checkpoint"] = str(checkpoint)
    config["model"]["allow_missing_checkpoint"] = False
    with pytest.raises(ValueError, match="class_order"):
        InferencePipeline(config)

from src.config import load_config, resolve_path


def test_default_config_loads():
    config = load_config("configs/default.yaml")
    assert config["dataset"]["sampling_rate"] == 360
    assert len(config["model"]["class_names"]) == 5
    assert resolve_path(config["model"]["checkpoint"]).name == "best_model.pt"

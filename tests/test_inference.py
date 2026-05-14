from src.config import load_config
from src.data.synthetic import synthetic_long_signal
from src.inference import InferencePipeline


def test_inference_output_schema():
    config = load_config("configs/default.yaml")
    result = InferencePipeline(config).predict_signal(synthetic_long_signal(seed=9), sampling_rate=360)
    assert result["status"] == "ok"
    assert result["prediction"]["class_name"] in config["model"]["class_names"]
    assert set(result["prediction"]["probabilities"]) == set(config["model"]["class_names"])
    assert result["input"]["window_count"] >= 1

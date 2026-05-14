from src.config import load_config
from src.data.dataset import leakage_report, load_dataset, split_dataset


def test_dataset_fallback_and_leakage_report():
    config = load_config("configs/default.yaml")
    bundle = load_dataset(config)
    splits = split_dataset(bundle, config)
    report = leakage_report(splits)
    assert bundle.X.shape[1] == config["preprocessing"]["window_size"]
    assert report["status"] == "passed"

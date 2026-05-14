from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from src.config import ensure_config_dirs, load_config, resolve_path
from src.data.dataset import load_dataset
from src.inference import InferencePipeline
from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run record-level grouped fold evaluation with no beat/window leakage.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_config_dirs(config)
    bundle = load_dataset(config)
    records = bundle.record_ids
    unique_records = sorted(set(records.tolist()))
    n_splits = min(args.folds, len(unique_records))
    pipeline = InferencePipeline({**config, "model": {**config["model"], "checkpoint": args.checkpoint or config["model"]["checkpoint"]}})
    out_dir = resolve_path(config["artifacts"].get("evaluation_dir", "artifacts/evaluation")) / "folds"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold_idx, (_, test_idx) in enumerate(splitter.split(bundle.X, bundle.y, groups=records), start=1):
        probs = pipeline.predict_windows(bundle.X[test_idx])
        preds = np.asarray([pipeline._predict_index(row) for row in probs], dtype=np.int64)
        y_true = bundle.y[test_idx]
        train_records = sorted(set(records[np.setdiff1d(np.arange(len(records)), test_idx)].tolist()))
        test_records = sorted(set(records[test_idx].tolist()))
        row = {
            "fold": fold_idx,
            "accuracy": float(accuracy_score(y_true, preds)),
            "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, preds, average="weighted", zero_division=0)),
            "test_records": test_records,
            "train_record_count": len(train_records),
            "test_record_count": len(test_records),
            "overlap_count": len(set(train_records) & set(test_records)),
        }
        rows.append(row)
        write_json(out_dir / f"fold_{fold_idx}.json", row)

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "fold_metrics.csv", index=False)
    aggregate = {
        "source": bundle.source,
        "real_mitbih": bundle.source == "mitbih",
        "folds": rows,
        "accuracy_mean": float(frame["accuracy"].mean()),
        "accuracy_std": float(frame["accuracy"].std(ddof=0)),
        "macro_f1_mean": float(frame["macro_f1"].mean()),
        "macro_f1_std": float(frame["macro_f1"].std(ddof=0)),
        "weighted_f1_mean": float(frame["weighted_f1"].mean()),
        "weighted_f1_std": float(frame["weighted_f1"].std(ddof=0)),
        "note": "This evaluates one trained checkpoint across grouped folds; use repeated training for full CV.",
    }
    write_json(out_dir / "aggregate_metrics.json", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()

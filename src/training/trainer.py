from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.config import resolve_path
from src.data.dataset import class_weights, leakage_report, load_dataset, split_dataset
from src.models import build_model
from src.utils.io import write_json
from src.utils.seeding import set_seed


def train_model(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    set_seed(int(training["seed"]))
    dataset = load_dataset(config)
    splits = split_dataset(dataset, config)

    class_names = list(config["model"]["class_names"])
    class_dist = {
        "train": _class_distribution(splits.y_train, class_names),
        "val": _class_distribution(splits.y_val, class_names),
        "test": _class_distribution(splits.y_test, class_names),
    }
    print(f"\nClass distribution:\n{json.dumps(class_dist, indent=2)}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = str(config["model"].get("type", "baseline_cnn"))
    model = build_model(model_type, num_classes=len(class_names), input_size=int(config["model"]["input_size"])).to(device)
    weights = None
    if training.get("class_weighting", True):
        class_weight_values = _training_class_weights(splits.y_train, len(class_names), training)
        weights = torch.tensor(class_weight_values, dtype=torch.float32, device=device)
        print(f"Class weights applied: {weights.cpu().numpy()}\n")

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    scheduler = _build_scheduler(optimizer, training)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training.get("mixed_precision", True)) and device.type == "cuda")
    sampler = (
        _weighted_sampler(splits.y_train, len(class_names), int(training["seed"]), training)
        if training.get("weighted_sampler", True)
        else None
    )
    train_loader = DataLoader(
        ECGWindowDataset(
            splits.X_train,
            splits.y_train,
            augment=bool(training.get("augmentation", {}).get("enabled", True)),
            augmentation_cfg=training.get("augmentation", {}),
            seed=int(training["seed"]),
        ),
        batch_size=int(training["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
    )
    val_x = torch.tensor(splits.X_val, dtype=torch.float32, device=device)
    val_y = torch.tensor(splits.y_val, dtype=torch.long, device=device)

    best_f1 = -1.0
    best_state = None
    patience = int(training.get("patience", 8))
    patience_count = 0
    history: list[dict[str, float]] = []
    started = time.time()

    for epoch in range(int(training["epochs"])):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            scaler.scale(loss).backward()
            if training.get("gradient_clip_norm"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * batch_x.size(0)

        val_loss, val_acc, val_f1 = _evaluate_torch_model(model, criterion, val_x, val_y)
        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, len(train_loader.dataset)),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "learning_rate": current_lr,
        }
        history.append(epoch_row)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_f1)
        elif scheduler is not None:
            scheduler.step()

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch + 1} (patience={patience_count})")
                break

    model_dir = resolve_path(config["artifacts"]["models_dir"])
    metrics_dir = resolve_path(config["artifacts"]["metrics_dir"])
    evaluation_dir = resolve_path(config["artifacts"].get("evaluation_dir", "artifacts/evaluation"))
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "best_model.pt"

    record_count = int(len(set([*splits.train_record_ids, *splits.val_record_ids, *splits.test_record_ids])))
    metrics_summary = {
        "best_val_macro_f1": best_f1,
        "epochs_ran": len(history),
        "final_val_macro_f1": history[-1]["val_macro_f1"] if history else None,
        "final_val_accuracy": history[-1]["val_accuracy"] if history else None,
    }
    checkpoint_payload = {
        "model_state_dict": best_state or model.state_dict(),
        "model_type": model_type,
        "class_order": class_names,
        "class_names": class_names,
        "class_mapping": config["dataset"]["class_mapping"],
        "input_size": int(config["model"]["input_size"]),
        "sampling_rate": int(config["dataset"]["sampling_rate"]),
        "preprocessing": {
            "window_size": int(config["preprocessing"]["window_size"]),
            "sampling_rate": int(config["dataset"]["sampling_rate"]),
            "lowcut_hz": float(config["preprocessing"]["lowcut_hz"]),
            "highcut_hz": float(config["preprocessing"]["highcut_hz"]),
            "normalize": bool(config["preprocessing"].get("normalize", True)),
        },
        "source": splits.source,
        "real_mitbih": splits.source == "mitbih",
        "git_commit": _git_commit(),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics_summary": metrics_summary,
        "best_val_macro_f1": best_f1,
        "epochs_ran": len(history),
    }
    torch.save(checkpoint_payload, checkpoint_path)

    manifest = leakage_report(splits)
    manifest.update(
        {
            "train_samples": int(len(splits.y_train)),
            "val_samples": int(len(splits.y_val)),
            "test_samples": int(len(splits.y_test)),
            "record_count": record_count,
            "class_order": class_names,
            "class_mapping": config["dataset"]["class_mapping"],
            "class_distribution": class_dist,
            "sampling_rate": int(config["dataset"]["sampling_rate"]),
            "split_strategy": config["dataset"]["split"].get("group_by", "random"),
        }
    )
    write_json(config["artifacts"]["split_manifest"], manifest)
    write_json(evaluation_dir / "splits.json", manifest)

    pd.DataFrame(history).to_csv(metrics_dir / "training_history.csv", index=False)
    _save_training_curves(metrics_dir / "training_curves.png", history)

    summary = {
        "status": "completed",
        "source": splits.source,
        "real_mitbih": splits.source == "mitbih",
        "record_count": record_count,
        "sampling_rate": int(config["dataset"]["sampling_rate"]),
        "split_strategy": config["dataset"]["split"].get("group_by", "random"),
        "model_type": model_type,
        "checkpoint": str(checkpoint_path),
        **metrics_summary,
        "duration_sec": round(time.time() - started, 4),
        "history": history,
        "warnings": splits.warnings,
        "disclaimer": config["project"]["disclaimer"],
        "class_distribution": class_dist,
        "training_controls": {
            "class_weighting": bool(training.get("class_weighting", True)),
            "weighted_sampler": bool(training.get("weighted_sampler", True)),
            "augmentation": training.get("augmentation", {}),
            "scheduler": training.get("scheduler"),
            "gradient_clip_norm": training.get("gradient_clip_norm"),
            "mixed_precision": bool(training.get("mixed_precision", True)) and device.type == "cuda",
            "early_stopping_metric": "val_macro_f1",
            "capped_training": bool(training.get("max_train_samples_per_class")) and not bool(training.get("uncapped")),
            "max_train_samples_per_class": training.get("max_train_samples_per_class"),
        },
    }
    write_json(metrics_dir / "training_summary.json", summary)
    return summary


class ECGWindowDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool,
        augmentation_cfg: dict[str, Any],
        seed: int,
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.augment = augment
        self.cfg = augmentation_cfg
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.X[idx].copy()
        if self.augment:
            window = _augment_window(window, self.cfg, self.rng)
        return torch.tensor(window, dtype=torch.float32), torch.tensor(int(self.y[idx]), dtype=torch.long)


def _evaluate_torch_model(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
) -> tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(val_x)
        val_loss = float(criterion(logits, val_y).item())
        val_pred = torch.argmax(logits, dim=1).cpu().numpy()
    val_true = val_y.cpu().numpy()
    return (
        val_loss,
        float(accuracy_score(val_true, val_pred)),
        float(f1_score(val_true, val_pred, average="macro", zero_division=0)),
    )


def _build_scheduler(optimizer: torch.optim.Optimizer, training: dict[str, Any]):
    scheduler_name = str(training.get("scheduler", "")).lower()
    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(training["epochs"])))
    return None


def _training_class_weights(y: np.ndarray, num_classes: int, training: dict[str, Any]) -> np.ndarray:
    weights = class_weights(y, num_classes=num_classes)
    max_weight = training.get("max_class_weight")
    if max_weight is not None:
        weights = np.minimum(weights, float(max_weight)).astype(np.float32)
    return weights


def _weighted_sampler(y: np.ndarray, num_classes: int, seed: int, training: dict[str, Any]) -> WeightedRandomSampler:
    weights = _training_class_weights(y, num_classes, training)
    power = float(training.get("sampler_weight_power", 1.0))
    if power != 1.0:
        weights = np.power(weights, power).astype(np.float32)
    sample_weights = torch.tensor([float(weights[int(label)]) for label in y], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True, generator=generator)


def _augment_window(window: np.ndarray, cfg: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    augmented = np.asarray(window, dtype=np.float32).copy()
    noise_std = float(cfg.get("gaussian_noise_std", 0.02))
    if noise_std > 0:
        augmented = augmented + rng.normal(0.0, noise_std, size=augmented.size).astype(np.float32)
    scale_min = float(cfg.get("amplitude_scale_min", 0.9))
    scale_max = float(cfg.get("amplitude_scale_max", 1.1))
    if scale_min > 0 and scale_max >= scale_min:
        augmented = augmented * float(rng.uniform(scale_min, scale_max))
    drift_std = float(cfg.get("baseline_drift_std", 0.02))
    if drift_std > 0:
        augmented = augmented + float(rng.normal(0.0, drift_std)) * np.sin(np.linspace(0, 2 * np.pi, augmented.size))
    shift_max = int(cfg.get("time_shift_max", 0))
    if shift_max > 0:
        augmented = np.roll(augmented, int(rng.integers(-shift_max, shift_max + 1)))
    max_abs = float(np.max(np.abs(augmented)))
    return (augmented / max_abs).astype(np.float32) if max_abs > 1e-8 else augmented.astype(np.float32)


def _save_training_curves(path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    plt.figure(figsize=(9, 4))
    plt.plot(frame["epoch"], frame["train_loss"], label="train loss")
    plt.plot(frame["epoch"], frame["val_loss"], label="val loss")
    plt.plot(frame["epoch"], frame["val_macro_f1"], label="val macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _class_distribution(y: np.ndarray, class_names: list[str]) -> dict[str, int]:
    return {name: int((y == idx).sum()) for idx, name in enumerate(class_names)}


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None

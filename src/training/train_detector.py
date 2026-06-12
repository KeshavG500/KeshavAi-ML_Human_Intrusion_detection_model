"""
train_detector.py
──────────────────
Fine-tune YOLOv8m on the intrusion detection dataset.
Supports COCO, MOT17, and VIRAT datasets.
Logs to MLflow and Weights & Biases.

Usage:
    python -m src.training.train_detector \
        --data data/processed/intrusion.yaml \
        --config src/training/config.yaml \
        --epochs 100
"""

import argparse
import os
import sys
from pathlib import Path

import mlflow
import torch
import yaml
from loguru import logger
from ultralytics import YOLO
from ultralytics.utils.callbacks.mlflow import callbacks as mlflow_callbacks

# ──────────────────────────── Dataset YAML Builder ────────────────────────

def create_dataset_yaml(
    train_path: str,
    val_path: str,
    test_path: str,
    output_path: str = "data/processed/intrusion.yaml",
) -> str:
    """Generate a YOLO-compatible dataset.yaml file."""
    dataset_config = {
        "path": str(Path(train_path).parent.parent.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 3,
        "names": {0: "person", 1: "vehicle", 2: "unknown"},
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    logger.info(f"Dataset YAML saved → {output_path}")
    return output_path


# ──────────────────────────── Training Loop ───────────────────────────────

def train(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg  = cfg["training"]
    aug_cfg    = cfg["augmentation"]

    logger.info(f"Starting training: {model_cfg['architecture']} | "
                f"epochs={args.epochs or train_cfg['epochs']} | "
                f"device={model_cfg['device']}")

    # ── MLflow setup ──────────────────────────────────────────────────────
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("intrusion-detection")

    # ── Load base model ───────────────────────────────────────────────────
    weights = train_cfg.get("pretrained_weights", f"{model_cfg['architecture']}.pt")
    model = YOLO(weights)
    logger.info(f"Base model loaded: {weights}")

    # ── Register MLflow callbacks ──────────────────────────────────────────
    if os.getenv("MLFLOW_TRACKING_URI"):
        for event, callback in mlflow_callbacks.items():
            model.add_callback(event, callback)

    # ── Training arguments ────────────────────────────────────────────────
    results = model.train(
        data=args.data,
        epochs=args.epochs or train_cfg["epochs"],
        batch=args.batch or train_cfg["batch_size"],
        imgsz=model_cfg["input_size"],
        device=model_cfg["device"],
        workers=train_cfg["workers"],
        optimizer=train_cfg["optimizer"],
        lr0=train_cfg["lr0"],
        lrf=train_cfg["lrf"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
        warmup_epochs=train_cfg["warmup_epochs"],
        patience=train_cfg["patience"],
        save_period=train_cfg["save_period"],
        project=train_cfg["project"],
        name=train_cfg["name"],
        seed=train_cfg["seed"],
        # Augmentation
        hsv_h=aug_cfg["hsv_h"],
        hsv_s=aug_cfg["hsv_s"],
        hsv_v=aug_cfg["hsv_v"],
        degrees=aug_cfg["degrees"],
        translate=aug_cfg["translate"],
        scale=aug_cfg["scale"],
        shear=aug_cfg["shear"],
        fliplr=aug_cfg["fliplr"],
        flipud=aug_cfg["flipud"],
        mosaic=aug_cfg["mosaic"],
        mixup=aug_cfg["mixup"],
        copy_paste=aug_cfg["copy_paste"],
        # Other
        val=True,
        verbose=True,
        amp=True,            # Automatic Mixed Precision (FP16 training)
        cache=True,          # Cache images in RAM for faster training
        close_mosaic=10,     # Disable mosaic last N epochs for stability
    )

    logger.success(f"Training complete! Best weights: {results.save_dir}/weights/best.pt")
    logger.info(f"Final mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    logger.info(f"Final mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

    # ── Export to ONNX ────────────────────────────────────────────────────
    if args.export_onnx:
        logger.info("Exporting to ONNX...")
        best_model = YOLO(f"{results.save_dir}/weights/best.pt")
        onnx_path = best_model.export(
            format="onnx",
            imgsz=model_cfg["input_size"],
            half=True,
            simplify=True,
            opset=17,
        )
        logger.success(f"ONNX model exported → {onnx_path}")

    return results


# ──────────────────────────── Validation ──────────────────────────────────

def validate(args):
    """Run validation on the best checkpoint."""
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=640, device="cuda")
    logger.info(f"\n{'─'*50}")
    logger.info(f"mAP@0.5:       {metrics.box.map50:.4f}")
    logger.info(f"mAP@0.5:0.95:  {metrics.box.map:.4f}")
    logger.info(f"Precision:     {metrics.box.mp:.4f}")
    logger.info(f"Recall:        {metrics.box.mr:.4f}")
    logger.info(f"{'─'*50}")
    return metrics


# ──────────────────────────── CLI ─────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for Human Intrusion Detection"
    )
    subparsers = parser.add_subparsers(dest="command")

    # train sub-command
    train_p = subparsers.add_parser("train", help="Run model training")
    train_p.add_argument("--data",        default="data/processed/intrusion.yaml")
    train_p.add_argument("--config",      default="src/training/config.yaml")
    train_p.add_argument("--epochs",      type=int, default=None)
    train_p.add_argument("--batch",       type=int, default=None)
    train_p.add_argument("--export-onnx", action="store_true")

    # validate sub-command
    val_p = subparsers.add_parser("validate", help="Validate a trained model")
    val_p.add_argument("--weights", required=True)
    val_p.add_argument("--data", default="data/processed/intrusion.yaml")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "validate":
        validate(args)
    else:
        logger.error("Use: python -m src.training.train_detector [train|validate]")
        sys.exit(1)

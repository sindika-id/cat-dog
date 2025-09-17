# train.py
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

import mlflow
import mlflow.pytorch

# ----------------------------
# Utilities
# ----------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, dict]:
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    train_data = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_data = datasets.ImageFolder(data_dir / "val", transform=val_tf)

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, train_data.class_to_idx


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch_idx: int,
    global_step: int,
    log_every: int,
) -> tuple[float, int]:
    """Returns (avg_loss, new_global_step)."""
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch_idx+1} · train", leave=False)
    for i, (imgs, labels) in enumerate(pbar, start=1):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running += loss.item()

        # Batch-level logging for line charts over step
        if log_every and (i % log_every == 0 or i == 1):
            mlflow.log_metric("train_loss_batch", float(loss.item()), step=global_step)
        global_step += 1

    avg_loss = running / max(1, len(loader))
    return avg_loss, global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    epoch_idx: int,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch_idx+1} · valid", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / max(1, len(loader))
    acc = correct / max(1, total)
    return avg_loss, acc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cat–Dog classifier training with MLflow tracking"
    )
    # Data / training
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Root folder containing train/ and val/")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights (avoids download)")
    p.add_argument("--log-every", type=int, default=10, help="Log batch loss every N steps (0=disable)")
    # Output
    p.add_argument("--out-dir", type=Path, default=Path("models"))
    p.add_argument("--save-best-only", action="store_true", help="Save only best checkpoint by val accuracy")
    # MLflow
    p.add_argument("--mlflow-tracking-uri", type=str, default="http://165.232.169.40:5000")
    p.add_argument("--mlflow-experiment", type=str, default=os.getenv("MLFLOW_EXPERIMENT_NAME", "CatDog"))
    p.add_argument("--mlflow-run-name", type=str, default=None)
    p.add_argument("--mlflow-register-as", type=str, default="catdog-v1.0", help="Optional registered model name")
    # S3/MinIO client defaults
    p.add_argument("--s3-endpoint", type=str,
                   default=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://165.232.169.40:9000"),
                   help="S3/MinIO endpoint URL for artifact store")
    p.add_argument("--aws-access-key-id", type=str,
                   default=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
                   help="AWS/MinIO access key (workshop default)")
    p.add_argument("--aws-secret-access-key", type=str,
                   default=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
                   help="AWS/MinIO secret key (workshop default)")
    p.add_argument("--aws-region", type=str,
                   default=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
                   help="AWS region (used by some SDKs)")
    return p.parse_args()


def _apply_s3_env_defaults(args) -> None:
    """Populate S3/MinIO-related env vars iff they are not already set."""
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", args.s3_endpoint)
    os.environ.setdefault("AWS_ACCESS_KEY_ID", args.aws_access_key_id)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", args.aws_secret_access_key)
    os.environ.setdefault("AWS_DEFAULT_REGION", args.aws_region)
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    os.environ.setdefault("AWS_S3_SIGNATURE_VERSION", "s3v4")


def main():
    args = parse_args()
    seed_everything(args.seed)
    _apply_s3_env_defaults(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloaders
    train_loader, val_loader, class_to_idx = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    num_classes = len(class_to_idx)

    # Model / loss / optim
    model = build_model(num_classes=num_classes, pretrained=(not args.no_pretrained))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Output dirs
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "catdog_model.pth"
    best_ckpt_path = out_dir / "catdog_model.best.pth"
    labelmap_path = out_dir / "class_to_idx.json"

    # MLflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    mlflow.enable_system_metrics_logging()

    run_tags = {"framework": "pytorch", "arch": "resnet18", "task": "cat-dog"}
    run_name = args.mlflow_run_name or f"resnet18_bs{args.batch_size}_lr{args.lr}"

    with mlflow.start_run(run_name=run_name, tags=run_tags):
        print("MLflow tracking URI:", mlflow.get_tracking_uri())
        print("MLflow artifact URI:", mlflow.get_artifact_uri())


        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_workers": args.num_workers,
            "pretrained": not args.no_pretrained,
            "seed": args.seed,
            "device": str(device),
            "log_every": args.log_every,
        })

        # Persist label map
        with open(labelmap_path, "w") as f:
            json.dump(class_to_idx, f, indent=2)
        mlflow.log_artifact(str(labelmap_path))

        best_acc = -1.0
        global_step = 0
        for epoch in range(args.epochs):
            t0 = time.time()

            # Train (batch-level logging inside)
            train_loss, global_step = train_one_epoch(
                model, train_loader, device, criterion, optimizer, epoch,
                global_step=global_step, log_every=args.log_every
            )

            # Validate
            val_loss, val_acc = evaluate(model, val_loader, device, criterion, epoch)
            epoch_ms = (time.time() - t0) * 1000.0

            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_acc={val_acc:.2%}  epoch_time_ms={epoch_ms:.1f}")

            # Epoch-level metrics (these drive per-run line charts when X=step)
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "epoch_time_ms": epoch_ms,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=epoch
            )

            # Save latest
            torch.save(model.state_dict(), ckpt_path)

            # Save best
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_ckpt_path)
                mlflow.log_artifact(str(best_ckpt_path))

        # Final artifacts
        mlflow.log_artifact(str(ckpt_path))

        # Log MLflow model
        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=args.mlflow_register_as if args.mlflow_register_as else None,
        )

        print(f"\nDone. Latest: {ckpt_path.name} | Best: {best_ckpt_path.name} (val_acc={best_acc:.2%})")
        print(f"Artifacts logged to MLflow experiment '{args.mlflow_experiment}' at {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    main()

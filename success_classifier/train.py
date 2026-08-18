"""Train ResNet18 success/fail classifier on prepared train.pt / val.pt.

Usage:
    .venv_success_cls/bin/python -m sim2real.components.success_classifier.train
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from sim2real.components.success_classifier.model import build_model
except ModuleNotFoundError:
    from success_classifier.model import build_model


class FrameDataset(Dataset):
    def __init__(self, pt_path: Path, image_size: int = 224) -> None:
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        self.images = payload["images"]
        self.labels = [float(v) for v in payload["labels"]]
        self.image_size = int(image_size)
        if len(self.images) != len(self.labels):
            raise ValueError(f"images/labels length mismatch in {pt_path}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Real (320x240) and teleop frames can differ; resize here so collate stacks.
        img = self.images[idx]
        if img.dtype != torch.uint8:
            img = img.to(torch.uint8)
        img_f = img.float().unsqueeze(0) / 255.0  # 1CHW in [0,1]
        img_f = F.interpolate(
            img_f,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        img = (img_f.squeeze(0) * 255.0).clamp(0, 255).to(torch.uint8)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(description="Train ResNet success classifier")
    p.add_argument(
        "--train-pt",
        type=Path,
        default=repo / "logs" / "success_classifier" / "processed" / "train.pt",
    )
    p.add_argument(
        "--val-pt",
        type=Path,
        default=repo / "logs" / "success_classifier" / "processed" / "val.pt",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run directory (default: logs/success_classifier/<timestamp>)",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional classifier checkpoint used to initialize fine-tuning.",
    )
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow training on CPU if CUDA is unavailable (slower)",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    tp = fp = tn = fn = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.numel()
        n += labels.numel()
        preds = (torch.sigmoid(logits) >= 0.5).long()
        y = labels.long()
        tp += int(((preds == 1) & (y == 1)).sum())
        fp += int(((preds == 1) & (y == 0)).sum())
        tn += int(((preds == 0) & (y == 0)).sum())
        fn += int(((preds == 0) & (y == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    acc = (tp + tn) / n if n else 0.0
    return {
        "loss": total_loss / n if n else 0.0,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
        print("[warn] CUDA unavailable — training on CPU")
    else:
        raise SystemExit(
            "CUDA is required. Pass --allow-cpu to train on CPU, "
            "or fix the NVIDIA driver / use a GPU machine."
        )

    repo = Path(__file__).resolve().parents[3]
    if args.output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = repo / "logs" / "success_classifier" / stamp
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    train_ds = FrameDataset(args.train_pt, image_size=args.image_size)
    val_ds = FrameDataset(args.val_pt, image_size=args.image_size)
    print(f"[data] train={len(train_ds)} val={len(val_ds)} device={device}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        image_size=args.image_size,
        pretrained=args.init_checkpoint is None,
    ).to(device)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(
            args.init_checkpoint, map_location=device, weights_only=False
        )
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state)
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    writer = SummaryWriter(log_dir=str(args.output_dir / "tb"))
    best_f1 = -1.0
    best_val_loss = float("inf")
    patience_left = args.patience
    history: list[dict] = []

    for epoch in range(1, args.max_epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.max_epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += float(loss.item()) * labels.numel()
            seen += labels.numel()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(seen, 1)
        val_metrics = evaluate(model, val_loader, device, criterion)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.3f} "
            f"prec={val_metrics['precision']:.3f} "
            f"rec={val_metrics['recall']:.3f} "
            f"f1={val_metrics['f1']:.3f}"
        )
        writer.add_scalar("train/loss", train_loss, epoch)
        for k, v in val_metrics.items():
            if k in ("loss", "accuracy", "precision", "recall", "f1"):
                writer.add_scalar(f"val/{k}", v, epoch)

        # Checkpoint every epoch + track best by F1 (tie-break: lower val loss).
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            },
            ckpt_dir / f"epoch_{epoch:03d}.pt",
        )

        improved = val_metrics["f1"] > best_f1 + args.min_delta or (
            abs(val_metrics["f1"] - best_f1) <= args.min_delta
            and val_metrics["loss"] < best_val_loss - args.min_delta
        )
        if improved:
            best_f1 = val_metrics["f1"]
            best_val_loss = val_metrics["loss"]
            patience_left = args.patience
            best_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_metrics": val_metrics,
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                },
                best_path,
            )
            print(f"[ckpt] new best → {best_path} (f1={best_f1:.3f})")
        else:
            patience_left -= 1
            print(f"[early-stop] no improvement, patience_left={patience_left}")
            if patience_left <= 0:
                print("[early-stop] stopping")
                break

    writer.close()
    metrics_path = args.output_dir / "history.json"
    metrics_path.write_text(json.dumps(history, indent=2))
    summary = {
        "best_f1": best_f1,
        "best_val_loss": best_val_loss,
        "best_checkpoint": str(ckpt_dir / "best.pt"),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "final": history[-1] if history else {},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()

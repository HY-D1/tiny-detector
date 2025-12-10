"""
Training loop entry point for TinyDetector.
"""

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import cfg
from .datasets import load_split
from .model import build_model
from .tokenization import get_tokenizer, make_collate_fn
from .utils import set_seed


def create_dataloaders():
    train_ds = load_split("train")
    val_ds = load_split("val")

    # Optionally subsample training data for quick debug
    if cfg.max_train_samples is not None and cfg.max_train_samples < len(train_ds):
        train_ds.texts = train_ds.texts[: cfg.max_train_samples]
        train_ds.labels = train_ds.labels[: cfg.max_train_samples]

    tokenizer = get_tokenizer()
    collate_fn = make_collate_fn(tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * batch["labels"].size(0)
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == batch["labels"]).sum().item()
            total_examples += batch["labels"].size(0)

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return avg_loss, accuracy


def train():
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders()

    model = build_model(num_labels=cfg.num_labels)
    model.to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    num_training_steps = cfg.num_epochs * len(train_loader)
    num_warmup_steps = int(cfg.warmup_ratio * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    best_val_accuracy = 0.0
    global_step = 0

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_examples = 0

        print(f"\nEpoch {epoch}/{cfg.num_epochs}")

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_size = batch["labels"].size(0)
            running_loss += loss.item() * batch_size
            running_examples += batch_size
            global_step += 1

            if step % 100 == 0 or step == 1:
                avg_loss = running_loss / running_examples
                print(f"  Step {step}/{len(train_loader)} - "
                      f"loss: {avg_loss:.4f}")

        # End of epoch: evaluate
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Validation - loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            save_path = cfg.checkpoints_dir / "best_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved to {save_path} "
                  f"(accuracy={best_val_accuracy:.4f})")


if __name__ == "__main__":
    train()

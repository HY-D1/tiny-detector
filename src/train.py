"""
Training entry point for TinyDetector.

Run:
  python -m src.train
"""

from collections import Counter
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup

from .config import cfg
from .datasets import load_split
from .model import build_model
from .tokenization import get_tokenizer, make_collate_fn
from .utils import set_seed


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency weights (normalized) for CrossEntropyLoss.
    """
    counts = Counter(labels)
    total = sum(counts.values())
    weights = []
    for c in range(num_classes):
        weights.append(total / max(counts.get(c, 1), 1))
    w = torch.tensor(weights, dtype=torch.float)
    w = w / w.sum() * num_classes
    return w


def make_train_loader(train_ds, tokenizer):
    collate_fn = make_collate_fn(tokenizer)

    if cfg.use_weighted_sampler:
        counts = Counter(train_ds.labels)
        class_weight = {c: 1.0 / counts[c] for c in counts}
        sample_weights = [class_weight[y] for y in train_ds.labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        return DataLoader(
            train_ds,
            batch_size=cfg.train_batch_size,
            sampler=sampler,
            collate_fn=collate_fn,
        )

    return DataLoader(
        train_ds,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )


def make_val_loader(val_ds, tokenizer):
    collate_fn = make_collate_fn(tokenizer)
    return DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    total_correct = 0
    total = 0
    total_loss = 0.0

    # unweighted loss for reporting
    ce = torch.nn.CrossEntropyLoss()

    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        logits = model(**batch).logits
        loss = ce(logits, labels)

        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    return total_loss / total, total_correct / total


def train():
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = load_split("train")
    val_ds = load_split("val")

    # optional debug subset
    if cfg.max_train_samples is not None and cfg.max_train_samples < len(train_ds):
        train_ds.texts = train_ds.texts[: cfg.max_train_samples]
        train_ds.labels = train_ds.labels[: cfg.max_train_samples]

    tokenizer = get_tokenizer()
    train_loader = make_train_loader(train_ds, tokenizer)
    val_loader = make_val_loader(val_ds, tokenizer)

    model = build_model(num_labels=cfg.num_labels).to(device)

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

    # weighted loss (helps minority class)
    if cfg.use_class_weighted_loss:
        class_w = compute_class_weights(train_ds.labels, cfg.num_labels).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_w)
        print("Using class-weighted loss:", class_w.detach().cpu().tolist())
    else:
        criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0

        print(f"\nEpoch {epoch}/{cfg.num_epochs}")

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            logits = model(**batch).logits
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_n += bs

            if step % 100 == 0 or step == 1:
                print(f"  Step {step}/{len(train_loader)} - loss: {running_loss / running_n:.4f}")

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Validation - loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = cfg.checkpoints_dir / "best_model.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved to {save_path} (accuracy={best_val_acc:.4f})")


if __name__ == "__main__":
    train()

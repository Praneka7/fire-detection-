"""Train a balanced EfficientNet fire/non-fire classifier from folder-labelled images."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms

IMAGE_SIZE = 224


def make_splits(targets: list[int], seed: int) -> tuple[list[int], list[int], list[int]]:
    """Create stratified 70/15/15 splits so every class appears in each set."""
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[target].append(index)
    rng = random.Random(seed)
    train, validation, test = [], [], []
    for indexes in by_class.values():
        rng.shuffle(indexes)
        count = len(indexes)
        train_end, val_end = int(count * 0.70), int(count * 0.85)
        train.extend(indexes[:train_end])
        validation.extend(indexes[train_end:val_end])
        test.extend(indexes[val_end:])
    return train, validation, test


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += criterion(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path, help="Folder containing fire_images and non-fire_images.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not use ImageNet initialization.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augment = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.35, contrast=0.25, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    normalise = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    source = datasets.ImageFolder(args.data_dir)
    if len(source.classes) != 2:
        raise SystemExit(f"Expected exactly two folders/classes, found: {source.classes}")
    train_ids, val_ids, test_ids = make_splits(source.targets, args.seed)
    train_dataset = Subset(datasets.ImageFolder(args.data_dir, augment), train_ids)
    eval_dataset = datasets.ImageFolder(args.data_dir, normalise)
    class_counts = torch.bincount(torch.tensor([source.targets[i] for i in train_ids]))
    sample_weights = [1.0 / class_counts[source.targets[i]].item() for i in train_ids]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(Subset(eval_dataset, val_ids), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(Subset(eval_dataset, test_ids), batch_size=args.batch_size, shuffle=False, num_workers=0)

    weights = None if args.no_pretrained else models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "fire_classifier.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs}: validation loss={val_loss:.4f}, accuracy={val_accuracy:.2%}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({"model_state": model.state_dict(), "class_to_idx": source.class_to_idx}, model_path)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    metrics = {"test_loss": test_loss, "test_accuracy": test_accuracy, "best_validation_accuracy": best_accuracy}
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {model_path}. Test accuracy: {test_accuracy:.2%}")


if __name__ == "__main__":
    main()

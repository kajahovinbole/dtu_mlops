import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from exercises_s2.model import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    print("Evaluating model:", model_checkpoint)

    # Load processed data
    images = torch.load("data/processed/images.pt")
    labels = torch.load("data/processed/labels.pt")

    dataset = TensorDataset(images, labels)

    # Load saved splits
    splits = torch.load("data/splits.pt")
    test_ds = Subset(dataset, splits["test"])
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # Load model
    model = Model().to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.eval()

    acc = MulticlassAccuracy(num_classes=10).to(device)
    f1 = MulticlassF1Score(num_classes=10, average="macro").to(device)

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            acc.update(out, y)
            f1.update(out, y)

    print(f"Test accuracy: {acc.compute().item():.4f}")
    print(f"Test F1 (macro): {f1.compute().item():.4f}")


if __name__ == "__main__":
    evaluate("models/best_model.pth")
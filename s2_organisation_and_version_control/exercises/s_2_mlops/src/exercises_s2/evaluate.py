import torch
from exercises_s2.model import Model
from exercises_s2.data import MyDataset
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    splits = torch.load("data/splits.pt")
    dataset = MyDataset("data/raw")
    test_ds = Subset(dataset, splits["test"])
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = Model().to(device)
    state = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state)

    acc = MulticlassAccuracy(num_classes=10).to(device)
    f1 = MulticlassF1Score(num_classes=10, average="macro").to(device)
    acc.reset()
    f1.reset()


    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            acc.update(out, y)
            f1.update(out, y)
        
    test_acc = acc.compute().item()
    test_f1 = f1.compute().item()

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test F1 (macro): {test_f1:.4f}")


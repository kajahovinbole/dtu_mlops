from exercises_s2.model import Model
from exercises_s2.data import MyDataset
import torch

from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
import matplotlib.pyplot as plt
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_accuracy = MulticlassAccuracy(num_classes=10).to(device)
train_f1 = MulticlassF1Score(num_classes=10, average="macro").to(device)
val_accuracy = MulticlassAccuracy(num_classes=10).to(device)
val_f1 = MulticlassF1Score(num_classes=10, average="macro").to(device)


splits = torch.load("data/splits.pt")
dataset = MyDataset("data/raw")

train_ds = Subset(dataset, splits["train"])
val_ds   = Subset(dataset, splits["val"])

lr = 0.001


def train():
    
    model = Model()
    model.to(device)
    

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 5

    # Early stopping:
    patience = 3
    best_val_loss = float("inf")
    patience_counter = 0

    train_loss_history = []
    train_acc_history = []
    train_f1_history = []

    val_loss_history = []
    val_acc_history = []
    val_f1_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_accuracy.reset()
        train_f1.reset()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x)
            batch_loss = criterion(out, y)

            batch_loss.backward()
            optimizer.step()
            
            train_loss_history.append(batch_loss.item())
            running_loss += batch_loss.item()

            with torch.no_grad():
                train_accuracy.update(out, y)
                train_f1.update(out, y)

        avg_loss = running_loss / len(train_loader)
        epoch_train_acc = train_accuracy.compute().item()
        epoch_train_f1 = train_f1.compute().item()

        train_acc_history.append(epoch_train_acc)
        train_f1_history.append(epoch_train_f1)


        # validation
        model.eval()
        val_running_loss = 0.0
        val_accuracy.reset()
        val_f1.reset()

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)

                val_loss = criterion(out, y)
                val_running_loss += val_loss.item()

                val_accuracy.update(out, y)
                val_f1.update(out, y)

        avg_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = val_accuracy.compute().item()
        epoch_val_f1 = val_f1.compute().item()

        val_loss_history.append(avg_val_loss)
        val_acc_history.append(epoch_val_acc)
        val_f1_history.append(epoch_val_f1)


        print(f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_loss:.4f} "
            f"Train Acc: {epoch_train_acc:.4f} "
            f"Train F1: {epoch_train_f1:.4f} "
            f"Val Loss: {avg_val_loss:.4f} "
            f"Val Acc: {epoch_val_acc:.4f} "
            f"Val F1: {epoch_val_f1:.4f}")
        
        # early stopping check
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    print("Finished Training")
    torch.save(model.state_dict(), MODELS_DIR / "model.pth")

    # Figs After training
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(train_loss_history)
    axs[0].set_title("Training loss")
    axs[0].set_xlabel("Step")
    axs[0].set_ylabel("Loss")

    axs[1].plot(train_acc_history)
    axs[1].set_title("Training accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_statistics.png")
    plt.close(fig)

    plt.figure(figsize=(6,4))
    plt.plot(train_f1_history)
    plt.title("Training F1 (macro)")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.savefig(FIG_DIR / "training_f1.png")
    plt.close()


if __name__ == "__main__":
    train()

import torch
import typer
from data import corrupt_mnist
from model import MyAwesomeModel
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
import matplotlib.pyplot as plt

model = MyAwesomeModel()
app = typer.Typer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

accuracy = MulticlassAccuracy(num_classes=10).to(device)
f1_score = MulticlassF1Score(num_classes=10, average = "macro").to(device)

@app.command()
def train(lr: float = 1e-3) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    model.train()
    X_train, y_train, _, _ = corrupt_mnist()

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs = 5

    train_loss_history = []
    train_acc_history = []
    train_f1_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        accuracy.reset()
        f1_score.reset()

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x)

            batch_loss = criterion(out, y)
            train_loss_history.append(batch_loss.item())

            batch_loss.backward()
            optimizer.step()

            running_loss += batch_loss.item()


            with torch.no_grad():
                accuracy.update(out, y)
                f1_score.update(out, y)

        avg_loss = running_loss / len(loader)
        train_acc = accuracy.compute().item()
        train_f1 = f1_score.compute().item()

        train_acc_history.append(train_acc)
        train_f1_history.append(train_f1)
        
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - accuracy: {train_acc:.4f} - f1: {train_f1:.4f}")


    print("Finished Training")
    torch.save(model.state_dict(), "model.pth")
  
    # etter trening
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
    fig.savefig("training_statistics.png")
    plt.close(fig)

    plt.figure(figsize=(6,4))
    plt.plot(train_f1_history)
    plt.title("Training F1 (macro)")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.savefig("training_f1.png")
    plt.close()




@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel().to(device)
    state = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state)

    _, _, X_test, y_test = corrupt_mnist()

    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

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



if __name__ == "__main__":
    app()

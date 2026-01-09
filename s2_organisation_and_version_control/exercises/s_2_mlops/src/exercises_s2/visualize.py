import matplotlib.pyplot as plt
import torch
import typer
from exercises_s2.model import Model
from sklearn.manifold import TSNE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize(model_checkpoint: str, figure_name: str = "embeddings.png") -> None:
    """Visualize embeddings from the trained model using t-SNE."""
    model = Model().to(DEVICE)
    state = torch.load(model_checkpoint, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    # Make the model output "features" instead of class logits.
    # This assumes your last layer is called fc2 (like in your model).
    # If your final layer has a different name, change this line.
    model.fc2 = torch.nn.Identity()

    images = torch.load("data/processed/images.pt")
    labels = torch.load("data/processed/labels.pt")
    dataset = torch.utils.data.TensorDataset(images, labels)

    embeddings, targets = [], []
    with torch.inference_mode():
        for x, y in torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False):
            x = x.to(DEVICE)
            feats = model(x)          # now returns features because fc2 is Identity
            embeddings.append(feats.cpu())
            targets.append(y)

    embeddings = torch.cat(embeddings).numpy()
    targets = torch.cat(targets).numpy()

    tsne = TSNE(n_components=2, random_state=42)
    emb2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(emb2d[mask, 0], emb2d[mask, 1], label=str(i), s=10)

    plt.legend()
    out_path = f"reports/figures/{figure_name}"
    plt.savefig(out_path, dpi=200)
    print(f"Saved t-SNE plot to {out_path}")


if __name__ == "__main__":
    typer.run(visualize)
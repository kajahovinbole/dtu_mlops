from pathlib import Path
import torch
import typer
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

app = typer.Typer()

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        datadir = Path(self.data_path)


        train_images, train_targets = [], []

        for f in sorted(datadir.glob("train_images_*.pt")):
            train_images.append(torch.load(f))

        for f in sorted(datadir.glob("train_target_*.pt")):
            train_targets.append(torch.load(f))

        X_imgs = torch.cat(train_images, dim=0)
        y_labels = torch.cat(train_targets, dim=0)

        X_test = torch.load(datadir / "test_images.pt")
        y_test = torch.load(datadir / "test_target.pt")


        imgs = torch.cat([X_imgs, X_test], dim=0)
        labels = torch.cat([y_labels, y_test], dim=0)

        imgs = imgs.unsqueeze(1)   # ← legger til channel-dim

        self.imgs = imgs
        self.labels = labels

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.imgs)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.imgs[index], self.labels[index]

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder.mkdir(parents=True, exist_ok=True)

        mean = self.imgs.mean()
        std = self.imgs.std()

        X_imgs_norm = (self.imgs - mean) / std

        torch.save(X_imgs_norm, output_folder / "images.pt")
        torch.save(self.labels, output_folder / "labels.pt")

        print(f"Saved processed data to {output_folder}")


@app.command()
def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)
    print(f"Number of samples in dataset: {len(dataset)}")



@app.command()
def inspect(processed_dir: Path) -> None:
    images = torch.load(processed_dir / "images.pt")
    labels = torch.load(processed_dir / "labels.pt")

    x, y = images[0], labels[0]
    plt.imshow(x.squeeze(), cmap="gray")
    plt.title(f"Label: {int(y)}")
    plt.axis("off")
    plt.show()
    


if __name__ == "__main__":
    app()
    


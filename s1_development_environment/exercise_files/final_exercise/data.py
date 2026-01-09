import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid


def corrupt_mnist():
    """Return train and test tensors for corrupt MNIST."""
    datadir = Path("data/corruptmnist_v1/")

    train_images = []
    train_targets= []

    for f in sorted(datadir.glob("train_images_*.pt")):
        data = torch.load(f)
        train_images.append(data)

    for f in sorted(datadir.glob("train_target_*.pt")):
        data = torch.load(f)
        train_targets.append(data)

    X_train = torch.cat(train_images, dim=0)
    y_train = torch.cat(train_targets, dim=0)

    X_test = torch.load(datadir / "test_images.pt")
    y_test = torch.load(datadir / "test_target.pt")

    # add channel dimension
    X_train = X_train.unsqueeze(1)
    X_test  = X_test.unsqueeze(1)

    return X_train, y_train, X_test, y_test

def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
    """Plot images and their labels in a grid."""
    row_col = int(len(images) ** 0.5)   # number of rows and columns
    fig = plt.figure(figsize=(10.0, 10.0)) # create figure
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3) # create grid
    for ax, im, label in zip(grid, images, target): # plot each image
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Label: {label.item()}")
        ax.axis("off")
    plt.show()


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = corrupt_mnist()
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Show some examples from the training set
    show_image_and_target(X_train[:16], y_train[:16])
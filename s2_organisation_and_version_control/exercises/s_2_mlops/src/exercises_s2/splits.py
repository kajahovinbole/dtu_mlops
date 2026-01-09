from pathlib import Path
import torch
from torch.utils.data import random_split

def make_splits(processed_dir: Path, out_path: Path, seed=42, train_frac=0.7, val_frac=0.15):
    images = torch.load(processed_dir / "images.pt")
    n = images.shape[0]

    train_size = int(train_frac * n)
    val_size = int(val_frac * n)
    test_size = n - train_size - val_size

    train_idx, val_idx, test_idx = random_split(
        range(n),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"train": list(train_idx), "val": list(val_idx), "test": list(test_idx)},
        out_path,
    )
    print(f"Saved splits to: {out_path.resolve()}")

if __name__ == "__main__":
    make_splits(Path("data/processed"), Path("data/splits.pt"))
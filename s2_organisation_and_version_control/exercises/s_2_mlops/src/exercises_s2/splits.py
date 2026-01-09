import torch
from torch.utils.data import random_split

def split(dataset, seed=42, train_frac=0.7, val_frac=0.15, test_frac=0.15, path="data/splits.pt"):
    """Splits the dataset into train, validation, and test sets and saves the indices"""
    n = len(dataset)
    train_size = int(train_frac * n)
    val_size = int(val_frac * n)
    test_size = n - train_size - val_size

    splits = random_split(
        range(n),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    split_indices = {
        "train": list(splits[0]),
        "val": list(splits[1]),
        "test": list(splits[2]),
    }

    torch.save(split_indices, path)
# from __future__ import annotations

# import pickle
# from pathlib import Path
# from dataclasses import dataclass
# from typing import Dict, List, Tuple

# import numpy as np

# try:
#     import torch
#     from torch.utils.data import Dataset
# except ImportError:
#     torch = None
#     Dataset = object


# # ============================================================
# # Config / container
# # ============================================================

# @dataclass
# class CIFARDataSplit:
#     X_initial: np.ndarray
#     y_initial: np.ndarray
#     X_pool: np.ndarray
#     y_pool: np.ndarray
#     X_test: np.ndarray
#     y_test: np.ndarray
#     class_names: List[str]


# # ============================================================
# # Low-level loading
# # ============================================================

# def _unpickle(file_path: str | Path) -> dict:
#     with open(file_path, "rb") as f:
#         return pickle.load(f, encoding="bytes")


# def _load_cifar_batch(batch_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Load one CIFAR-10 batch file.

#     Returns
#     -------
#     X : np.ndarray of shape (N, 32, 32, 3), dtype uint8
#     y : np.ndarray of shape (N,)
#     """
#     batch = _unpickle(batch_path)

#     X = batch[b"data"]                     # shape (N, 3072)
#     y = np.array(batch[b"labels"])         # shape (N,)

#     # Reshape from flat to image format
#     X = X.reshape(-1, 3, 32, 32)           # (N, C, H, W)
#     X = np.transpose(X, (0, 2, 3, 1))      # (N, H, W, C)

#     return X, y


# def _load_class_names(meta_path: str | Path) -> List[str]:
#     meta = _unpickle(meta_path)
#     names = meta[b"label_names"]
#     return [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in names]


# def load_cifar10_from_raw(data_dir: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
#     """
#     Load CIFAR-10 from the raw 'cifar-10-batches-py' folder.

#     Expected files:
#         data_batch_1 ... data_batch_5
#         test_batch
#         batches.meta
#     """
#     data_dir = Path(data_dir)

#     train_images = []
#     train_labels = []

#     for i in range(1, 6):
#         batch_path = data_dir / f"data_batch_{i}"
#         X_batch, y_batch = _load_cifar_batch(batch_path)
#         train_images.append(X_batch)
#         train_labels.append(y_batch)

#     X_train = np.concatenate(train_images, axis=0)
#     y_train = np.concatenate(train_labels, axis=0)

#     X_test, y_test = _load_cifar_batch(data_dir / "test_batch")
#     class_names = _load_class_names(data_dir / "batches.meta")

#     return X_train, y_train, X_test, y_test, class_names


# # ============================================================
# # Stratified initial split for active learning
# # ============================================================

# def stratified_initial_split(
#     X: np.ndarray,
#     y: np.ndarray,
#     n_initial_per_class: int = 10,
#     seed: int = 42,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Return indices for:
#       - initial labeled set
#       - unlabeled pool

#     Ensures exactly n_initial_per_class samples from each class
#     in the initial labeled set.
#     """
#     rng = np.random.default_rng(seed)

#     initial_indices = []
#     pool_indices = []

#     classes = np.unique(y)

#     for cls in classes:
#         cls_idx = np.where(y == cls)[0]
#         rng.shuffle(cls_idx)

#         if len(cls_idx) < n_initial_per_class:
#             raise ValueError(
#                 f"Class {cls} has only {len(cls_idx)} samples, "
#                 f"cannot take {n_initial_per_class} initial samples."
#             )

#         initial_cls = cls_idx[:n_initial_per_class]
#         pool_cls = cls_idx[n_initial_per_class:]

#         initial_indices.extend(initial_cls.tolist())
#         pool_indices.extend(pool_cls.tolist())

#     initial_indices = np.array(initial_indices)
#     pool_indices = np.array(pool_indices)

#     rng.shuffle(initial_indices)
#     rng.shuffle(pool_indices)

#     return initial_indices, pool_indices


# def prepare_data(
#     data_dir: str | Path = "data/cifar-10-batches-py",
#     n_initial_per_class: int = 10,
#     seed: int = 42,
# ) -> CIFARDataSplit:
#     """
#     Load CIFAR-10 and split training set into:
#       - initial labeled set
#       - unlabeled pool

#     The official CIFAR-10 test set is kept unchanged.
#     """
#     X_train, y_train, X_test, y_test, class_names = load_cifar10_from_raw(data_dir)

#     initial_idx, pool_idx = stratified_initial_split(
#         X_train,
#         y_train,
#         n_initial_per_class=n_initial_per_class,
#         seed=seed,
#     )

#     return CIFARDataSplit(
#         X_initial=X_train[initial_idx],
#         y_initial=y_train[initial_idx],
#         X_pool=X_train[pool_idx],
#         y_pool=y_train[pool_idx],
#         X_test=X_test,
#         y_test=y_test,
#         class_names=class_names,
#     )


# # ============================================================
# # Optional PyTorch dataset wrapper
# # ============================================================

# class CIFARDataset(Dataset):
#     def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
#         self.X = X
#         self.y = y
#         self.transform = transform

#     def __len__(self) -> int:
#         return len(self.X)

#     def __getitem__(self, idx: int):
#         image = self.X[idx]
#         label = int(self.y[idx])

#         if self.transform is not None:
#             image = self.transform(image)
#         elif torch is not None:
#             image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

#             # Normalize
#             mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
#             std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

#             image = (image - mean) / std

#         return image, label


# # ============================================================
# # Utility
# # ============================================================

# def print_split_summary(data: CIFARDataSplit) -> None:
#     print("Class names:", data.class_names)
#     print()
#     print(f"Initial labeled set: {len(data.X_initial)}")
#     print(f"Unlabeled pool:      {len(data.X_pool)}")
#     print(f"Test set:            {len(data.X_test)}")
#     print()

#     print("Initial labeled class counts:")
#     unique, counts = np.unique(data.y_initial, return_counts=True)
#     for u, c in zip(unique, counts):
#         print(f"  {data.class_names[u]}: {c}")


# # ============================================================
# # Example usage
# # ============================================================

# if __name__ == "__main__":
#     data = prepare_data(
#         data_dir="data/cifar-10-batches-py",
#         n_initial_per_class=10,   # try 20 for a slightly larger start
#         seed=42,
#     )

#     print_split_summary(data)

from __future__ import annotations

import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
except ImportError:
    torch = None
    Dataset = object
    transforms = None


# ============================================================
# Config / container
# ============================================================

@dataclass
class CIFARDataSplit:
    X_initial: np.ndarray
    y_initial: np.ndarray
    X_pool: np.ndarray
    y_pool: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: List[str]


# ============================================================
# Transforms
# ============================================================

def get_train_transform():
    """
    Transform for labeled training data.

    Since you use pretrained ImageNet weights, we:
    - convert to PIL
    - resize to 224x224
    - convert to tensor
    - normalize with ImageNet mean/std

    RandomHorizontalFlip is included as mild augmentation.
    """
    if transforms is None:
        raise ImportError("torchvision is required for transforms.")

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_eval_transform():
    """
    Transform for pool/test data.

    No random augmentation here, because:
    - pool predictions should be stable for active learning
    - test evaluation should be deterministic
    """
    if transforms is None:
        raise ImportError("torchvision is required for transforms.")

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# ============================================================
# Low-level loading
# ============================================================

def _unpickle(file_path: str | Path) -> dict:
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _load_cifar_batch(batch_path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load one CIFAR-10 batch file.

    Returns
    -------
    X : np.ndarray of shape (N, 32, 32, 3), dtype uint8
    y : np.ndarray of shape (N,)
    """
    batch = _unpickle(batch_path)

    X = batch[b"data"]              # shape (N, 3072)
    y = np.array(batch[b"labels"])  # shape (N,)

    # Reshape from flat vectors to image format
    X = X.reshape(-1, 3, 32, 32)    # (N, C, H, W)
    X = np.transpose(X, (0, 2, 3, 1))  # (N, H, W, C)

    return X, y


def _load_class_names(meta_path: str | Path) -> List[str]:
    meta = _unpickle(meta_path)
    names = meta[b"label_names"]
    return [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in names]


def load_cifar10_from_raw(
    data_dir: str | Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load CIFAR-10 from raw 'cifar-10-batches-py' folder.

    Expected files:
        data_batch_1 ... data_batch_5
        test_batch
        batches.meta
    """
    data_dir = Path(data_dir)

    train_images = []
    train_labels = []

    for i in range(1, 6):
        batch_path = data_dir / f"data_batch_{i}"
        X_batch, y_batch = _load_cifar_batch(batch_path)
        train_images.append(X_batch)
        train_labels.append(y_batch)

    X_train = np.concatenate(train_images, axis=0)
    y_train = np.concatenate(train_labels, axis=0)

    X_test, y_test = _load_cifar_batch(data_dir / "test_batch")
    class_names = _load_class_names(data_dir / "batches.meta")

    return X_train, y_train, X_test, y_test, class_names


# ============================================================
# Stratified initial split for active learning
# ============================================================

def stratified_initial_split(
    X: np.ndarray,
    y: np.ndarray,
    n_initial_per_class: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return indices for:
      - initial labeled set
      - unlabeled pool

    Ensures exactly n_initial_per_class samples from each class
    in the initial labeled set.
    """
    rng = np.random.default_rng(seed)

    initial_indices = []
    pool_indices = []

    classes = np.unique(y)

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)

        if len(cls_idx) < n_initial_per_class:
            raise ValueError(
                f"Class {cls} has only {len(cls_idx)} samples, "
                f"cannot take {n_initial_per_class} initial samples."
            )

        initial_cls = cls_idx[:n_initial_per_class]
        pool_cls = cls_idx[n_initial_per_class:]

        initial_indices.extend(initial_cls.tolist())
        pool_indices.extend(pool_cls.tolist())

    initial_indices = np.array(initial_indices)
    pool_indices = np.array(pool_indices)

    rng.shuffle(initial_indices)
    rng.shuffle(pool_indices)

    return initial_indices, pool_indices


def prepare_data(
    data_dir: str | Path = "data/cifar-10-batches-py",
    n_initial_per_class: int = 10,
    seed: int = 42,
) -> CIFARDataSplit:
    """
    Load CIFAR-10 and split training set into:
      - initial labeled set
      - unlabeled pool

    The official CIFAR-10 test set is kept unchanged.
    """
    X_train, y_train, X_test, y_test, class_names = load_cifar10_from_raw(data_dir)

    initial_idx, pool_idx = stratified_initial_split(
        X_train,
        y_train,
        n_initial_per_class=n_initial_per_class,
        seed=seed,
    )

    return CIFARDataSplit(
        X_initial=X_train[initial_idx],
        y_initial=y_train[initial_idx],
        X_pool=X_train[pool_idx],
        y_pool=y_train[pool_idx],
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
    )


# ============================================================
# PyTorch dataset wrapper
# ============================================================

class CIFARDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        image = self.X[idx]
        label = int(self.y[idx])

        if self.transform is not None:
            image = self.transform(image)
        else:
            if torch is None:
                raise ImportError("PyTorch is required when no transform is provided.")

            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, label


# ============================================================
# Helper to build datasets directly
# ============================================================

def build_datasets(
    data_dir: str | Path = "data/cifar-10-batches-py",
    n_initial_per_class: int = 10,
    seed: int = 42,
):
    """
    Prepare split and return:
      - split info
      - initial labeled dataset
      - unlabeled pool dataset
      - test dataset
    """
    split = prepare_data(
        data_dir=data_dir,
        n_initial_per_class=n_initial_per_class,
        seed=seed,
    )

    train_transform = get_train_transform()
    eval_transform = get_eval_transform()

    initial_dataset = CIFARDataset(
        split.X_initial,
        split.y_initial,
        transform=train_transform,
    )

    pool_dataset = CIFARDataset(
        split.X_pool,
        split.y_pool,
        transform=eval_transform,
    )

    test_dataset = CIFARDataset(
        split.X_test,
        split.y_test,
        transform=eval_transform,
    )

    return split, initial_dataset, pool_dataset, test_dataset


# ============================================================
# Utility
# ============================================================

def print_split_summary(data: CIFARDataSplit) -> None:
    print("Class names:", data.class_names)
    print()
    print(f"Initial labeled set: {len(data.X_initial)}")
    print(f"Unlabeled pool:      {len(data.X_pool)}")
    print(f"Test set:            {len(data.X_test)}")
    print()

    print("Initial labeled class counts:")
    unique, counts = np.unique(data.y_initial, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {data.class_names[u]}: {c}")


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    split, initial_dataset, pool_dataset, test_dataset = build_datasets(
        data_dir="data/cifar-10-batches-py",
        n_initial_per_class=10,
        seed=42,
    )

    print_split_summary(split)

    print()
    print("Dataset sizes:")
    print("Initial dataset:", len(initial_dataset))
    print("Pool dataset:   ", len(pool_dataset))
    print("Test dataset:   ", len(test_dataset))
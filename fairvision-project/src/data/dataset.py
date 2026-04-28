"""
FairVision - Data Loading and Preprocessing
============================================
Handles:
- Loading FairFace from HuggingFace (0.25 config)
- Train / internal-val / test splitting
- PyTorch Dataset wrapper
- Transforms (train augmentation + val/test normalization)
- WeightedRandomSampler for balanced mini-batch mitigation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from collections import Counter
from PIL import Image
from typing import List, Optional, Tuple

# ── Label names ────────────────────────────────────────────────────────────────
AGE_NAMES = [
    "0-2", "3-9", "10-19", "20-29", "30-39",
    "40-49", "50-59", "60-69", "70+"
]
GENDER_NAMES = ["Male", "Female"]
RACE_NAMES   = [
    "East Asian", "Indian", "Black", "White",
    "Middle Eastern", "Latino_Hispanic", "Southeast Asian"
]
NUM_AGE_CLASSES = len(AGE_NAMES)

# ── ImageNet stats (used for normalisation) ────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ── Transforms ─────────────────────────────────────────────────────────────────
def get_train_transform() -> transforms.Compose:
    """Augmented transform for training split."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_val_test_transform() -> transforms.Compose:
    """Deterministic transform for validation and test splits."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


# ── PyTorch Dataset ─────────────────────────────────────────────────────────────
class FairFaceDataset(Dataset):
    """
    Wraps a HuggingFace FairFace split into a PyTorch Dataset.

    Args:
        hf_dataset : HuggingFace dataset split object
        indices    : Subset indices (None = use all)
        transform  : torchvision transform to apply to each image
    """

    def __init__(self, hf_dataset, indices: Optional[List[int]] = None,
                 transform=None):
        self.hf_dataset = hf_dataset
        self.indices    = indices if indices is not None else list(range(len(hf_dataset)))
        self.transform  = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int]:
        real_idx = self.indices[idx]
        sample   = self.hf_dataset[real_idx]
        image    = sample["image"].convert("RGB")
        age      = int(sample["age"])
        gender   = int(sample["gender"])
        race     = int(sample["race"])
        if self.transform:
            image = self.transform(image)
        return image, age, gender, race


# ── Dataset loading ─────────────────────────────────────────────────────────────
def load_fairface(seed: int = 42, val_ratio: float = 0.2):
    """
    Load FairFace (0.25 config) and split into:
      - internal train (80% of HF train)
      - internal val   (20% of HF train)
      - final test     (HF validation split — held-out)

    Returns:
        hf_train, hf_test, train_idx, val_idx
    """
    print("Loading FairFace dataset (config=0.25) from HuggingFace...")
    dataset  = load_dataset("HuggingFaceM4/FairFace", "0.25")
    hf_train = dataset["train"]       # 86,744 samples
    hf_test  = dataset["validation"]  # 10,954 samples

    all_idx = list(range(len(hf_train)))
    age_labels = hf_train["age"]

    train_idx, val_idx = train_test_split(
        all_idx,
        test_size=val_ratio,
        random_state=seed,
        stratify=age_labels      # stratified to preserve age distribution
    )

    print(f"  Internal train : {len(train_idx):,}")
    print(f"  Internal val   : {len(val_idx):,}")
    print(f"  Final test     : {len(hf_test):,}")
    return hf_train, hf_test, train_idx, val_idx


# ── Class weights ───────────────────────────────────────────────────────────────
def compute_class_weights(age_labels: List[int],
                           num_classes: int = NUM_AGE_CLASSES) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.
    Rarer classes get higher weights.
    """
    counts  = np.bincount(age_labels, minlength=num_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes   # normalize
    return torch.tensor(weights, dtype=torch.float)


# ── DataLoader factory ──────────────────────────────────────────────────────────
def build_dataloaders(hf_train, hf_test, train_idx: List[int], val_idx: List[int],
                      batch_size: int = 64, num_workers: int = 2,
                      balanced_sampler: bool = False):
    """
    Build train / val / test DataLoaders.

    Args:
        balanced_sampler : If True, use WeightedRandomSampler on train loader
                           (Mitigation Strategy 2).
    Returns:
        train_loader, val_loader, test_loader, class_weights_tensor
    """
    train_transform = get_train_transform()
    val_transform   = get_val_test_transform()

    train_ds = FairFaceDataset(hf_train, train_idx, train_transform)
    val_ds   = FairFaceDataset(hf_train, val_idx,   val_transform)
    test_ds  = FairFaceDataset(hf_test,  None,       val_transform)

    # Class weights (used for weighted loss or sampler)
    train_age_labels = [int(hf_train[i]["age"]) for i in train_idx]
    class_weights    = compute_class_weights(train_age_labels)

    # Sampler
    if balanced_sampler:
        sample_w = torch.tensor(
            [class_weights[label].item() for label in train_age_labels],
            dtype=torch.float
        )
        sampler      = WeightedRandomSampler(sample_w, num_samples=len(train_idx), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    return train_loader, val_loader, test_loader, class_weights

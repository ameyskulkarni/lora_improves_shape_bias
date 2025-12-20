"""Dataloaders for ImageNet variants."""
import os
from pathlib import Path
from typing import Optional, Callable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image


class ImageNetV2Dataset(Dataset):
    """ImageNet-V2 with correct label mapping.

    ImageNet-V2 folders are named 0-999 (as strings), but ImageFolder
    sorts alphabetically: 0, 1, 10, 100, ... which breaks label mapping.
    """

    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'}

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        # Iterate folders in numeric order
        for class_idx in range(1000):
            class_dir = self.root / str(class_idx)
            if class_dir.exists():
                for img_path in class_dir.iterdir():
                    if img_path.suffix in self.VALID_EXTENSIONS:
                        self.samples.append((str(img_path), class_idx))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in {root}. "
                f"Expected structure: {root}/0/, {root}/1/, ..., {root}/999/ "
                f"with image files inside each folder."
            )
        print(f"ImageNetV2Dataset: found {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "labels": label}


class ImageNetDataset(Dataset):
    """Unified dataset for ImageNet variants."""

    VARIANTS = ["imagenet", "stylized_imagenet", "imagenet_c", "cue_conflict", "imagenetv2", "imagenet_sketch"]

    def __init__(
        self,
        root: str,
        variant: str = "imagenet",
        split: str = "train",
        transform: Optional[Callable] = None,
        corruption_type: Optional[str] = None,
        severity: int = 3,
    ):
        self.variant = variant
        self.transform = transform
        self._v2_dataset = None
        self.dataset = None

        # Handle ImageNet-V2 specially due to folder naming
        if variant == "imagenetv2":
            data_path = Path(root) / "imagenet-v2"/"imagenetv2-matched-frequency-val"/ split
            if not data_path.exists():
                raise RuntimeError(
                    f"ImageNet-V2 not found at {data_path}. "
                    f"Download from https://github.com/modestyachts/ImageNetV2 "
                    f"and ensure folder is named 'imagenetv2-matched-frequency-format-val'"
                )
            self._v2_dataset = ImageNetV2Dataset(str(data_path), transform)
            self.classes = [str(i) for i in range(1000)]
            self.class_to_idx = {str(i): i for i in range(1000)}
            return

        # Build path based on variant
        if variant == "imagenet":
            data_path = Path(root) / "imagenet" / split
        elif variant == "stylized_imagenet":
            data_path = Path(root) / "stylized_imagenet" / split
        elif variant == "imagenet_c":
            assert corruption_type, "Specify corruption_type for ImageNet-C"
            data_path = Path(root) / "imagenet-c" / corruption_type / str(severity)
        elif variant == "cue_conflict":
            data_path = Path(root) / "cue-conflict"
        elif variant == "imagenet_sketch":
            data_path = Path(root) / "imagenet-sketch" / "sketch"
        else:
            raise ValueError(f"Unknown variant: {variant}")

        self.dataset = ImageFolder(str(data_path), transform=None)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        if self._v2_dataset is not None:
            return len(self._v2_dataset)
        return len(self.dataset)

    def __getitem__(self, idx):
        if self._v2_dataset is not None:
            return self._v2_dataset[idx]

        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return {"pixel_values": img, "labels": label}


def get_transforms(processor, is_train: bool = True):
    """Get transforms compatible with ViT/DeiT processor.

    Uses BICUBIC interpolation and correct ImageNet normalization.
    Note: HuggingFace DeiT processor has wrong mean/std ([0.5,0.5,0.5]),
    so we override with correct ImageNet values.
    """
    # DeiT uses ImageNet normalization, NOT [0.5, 0.5, 0.5]
    # HuggingFace processor has wrong values, so we hardcode correct ones
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # Handle different processor.size formats
    if isinstance(processor.size, dict):
        if "height" in processor.size:
            size = processor.size["height"]
        elif "shortest_edge" in processor.size:
            size = processor.size["shortest_edge"]
        else:
            size = 224
    elif isinstance(processor.size, int):
        size = processor.size
    else:
        size = 224

    # DeiT uses crop_pct=0.875, so eval resize = 224/0.875 = 256
    eval_resize = int(size / 0.875)  # 256 for size=224

    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(eval_resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])


def create_dataloaders(
    data_root: str,
    processor,
    variant: str = "stylized_imagenet",
    batch_size: int = 64,
    num_workers: int = 4,
    **kwargs,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders (same variant for both)."""
    train_ds = ImageNetDataset(
        data_root, variant=variant, split="train",
        transform=get_transforms(processor, is_train=True), **kwargs
    )
    val_ds = ImageNetDataset(
        data_root, variant=variant, split="val",
        transform=get_transforms(processor, is_train=False), **kwargs
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def create_dataloaders_mixed(
    data_root: str,
    processor,
    train_variant: str = "stylized_imagenet",
    val_variant: str = "imagenet",
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders with different variants.

    Typical use: train on stylized_imagenet, validate on normal ImageNet.
    """
    train_ds = ImageNetDataset(
        data_root, variant=train_variant, split="train",
        transform=get_transforms(processor, is_train=True),
    )
    val_ds = ImageNetDataset(
        data_root, variant=val_variant, split="val",
        transform=get_transforms(processor, is_train=False),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
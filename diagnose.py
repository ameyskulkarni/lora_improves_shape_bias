"""Diagnostic script to verify data loading and model accuracy."""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor
from torchvision import transforms
from torchvision.datasets import ImageFolder


def test_raw_model(data_root: str, model_name: str = "facebook/deit-tiny-patch16-224"):
    """Test raw HuggingFace model without any wrappers."""
    print(f"\n=== Testing Raw Model: {model_name} ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model directly
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    model.to(device).eval()

    print(f"\nProcessor config:")
    print(f"  size: {processor.size}")
    print(f"  image_mean: {processor.image_mean}")
    print(f"  image_std: {processor.image_std}")
    print(f"  resample: {processor.resample}")

    # Parse size correctly
    if isinstance(processor.size, dict):
        if "height" in processor.size:
            size = processor.size["height"]
        elif "shortest_edge" in processor.size:
            size = processor.size["shortest_edge"]
        else:
            size = 224
    else:
        size = 224

    eval_resize = int(size / 0.875)  # 256 for DeiT
    print(f"  Computed size: {size}, eval_resize: {eval_resize}")

    # DeiT uses ImageNet normalization, NOT [0.5, 0.5, 0.5] from HF processor
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    print(f"\nUsing CORRECT ImageNet normalization:")
    print(f"  mean: {IMAGENET_MEAN}")
    print(f"  std: {IMAGENET_STD}")

    # Standard DeiT eval transforms
    transform = transforms.Compose([
        transforms.Resize(eval_resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Load ImageNet val
    imagenet_path = Path(data_root) / "imagenet" / "val"
    if not imagenet_path.exists():
        print(f"ERROR: ImageNet val not found at {imagenet_path}")
        return

    dataset = ImageFolder(str(imagenet_path), transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\nDataset size: {len(dataset)}")
    print(f"Running full evaluation...")

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.logits.argmax(-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"\n{'='*50}")
    print(f"Raw model accuracy: {acc*100:.2f}%")
    print(f"Expected for DeiT-Tiny: 72.2%")
    print(f"{'='*50}")

    if acc < 0.70:
        print("\nWARNING: Accuracy significantly below expected!")
        print("Possible issues:")
        print("  1. Wrong ImageNet val set (should have 50,000 images)")
        print("  2. Label mapping issue (check class folder names)")
        print("  3. Image corruption")
    elif acc < 0.72:
        print("\nSlightly below expected - might be minor preprocessing difference")
    else:
        print("\nAccuracy looks correct!")


def check_imagenet_structure(data_root: str):
    """Verify ImageNet structure."""
    print("\n=== Checking ImageNet Structure ===")

    val_path = Path(data_root) / "imagenet" / "val"
    if not val_path.exists():
        print(f"ERROR: {val_path} does not exist")
        return

    classes = sorted(os.listdir(val_path))
    print(f"Number of classes: {len(classes)}")
    print(f"First 5 classes: {classes[:5]}")
    print(f"Last 5 classes: {classes[-5:]}")

    # Count total images
    total_images = 0
    for cls in classes:
        cls_path = val_path / cls
        if cls_path.is_dir():
            total_images += len(list(cls_path.iterdir()))

    print(f"Total images: {total_images}")
    print(f"Expected: 50,000")

    if total_images != 50000:
        print("WARNING: Image count mismatch!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="facebook/deit-tiny-patch16-224")
    args = parser.parse_args()

    check_imagenet_structure(args.data_root)
    test_raw_model(args.data_root, args.model_name)


if __name__ == "__main__":
    main()
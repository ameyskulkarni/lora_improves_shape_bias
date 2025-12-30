#!/usr/bin/env python
"""Test and visualize depth dataset integration.

Usage:
    python test_depth_dataset.py --data-root /path/to/data --num-samples 8
"""
import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def test_depth_dataset(data_root: str, num_samples: int = 8, save_path: str = None):
    """Test that depth dataset loads correctly and visualize samples."""

    # Import after argparse to allow --help without full imports
    from datasets import (
        ImageNetDepthDataset,
        ImageNetDataset,
        get_transforms,
        verify_depth_dataset,
    )
    from transformers import ViTImageProcessor

    print("=" * 60)
    print("DEPTH DATASET VERIFICATION")
    print("=" * 60)

    # 1. Verify alignment
    print("\n1. Checking dataset alignment...")
    verify_depth_dataset(data_root, split="val")

    # 2. Load both datasets
    print("\n2. Loading datasets...")
    processor = ViTImageProcessor.from_pretrained("facebook/deit-tiny-patch16-224")
    transform = get_transforms(processor, is_train=False)

    try:
        rgb_dataset = ImageNetDataset(
            data_root, variant="imagenet", split="val", transform=transform
        )
        depth_dataset = ImageNetDepthDataset(
            data_root, split="val", transform=transform
        )
        print(f"   RGB dataset: {len(rgb_dataset)} samples")
        print(f"   Depth dataset: {len(depth_dataset)} samples")
    except Exception as e:
        print(f"   ERROR loading datasets: {e}")
        return

    # 3. Check sample statistics
    print("\n3. Sample statistics...")

    # Get a few RGB samples
    rgb_samples = [rgb_dataset[i] for i in range(min(100, len(rgb_dataset)))]
    rgb_tensors = torch.stack([s["pixel_values"] for s in rgb_samples])

    # Get a few depth samples
    depth_samples = [depth_dataset[i] for i in range(min(100, len(depth_dataset)))]
    depth_tensors = torch.stack([s["pixel_values"] for s in depth_samples])

    print(f"   RGB - mean: {rgb_tensors.mean():.4f}, std: {rgb_tensors.std():.4f}")
    print(f"   RGB - min: {rgb_tensors.min():.4f}, max: {rgb_tensors.max():.4f}")
    print(f"   Depth - mean: {depth_tensors.mean():.4f}, std: {depth_tensors.std():.4f}")
    print(f"   Depth - min: {depth_tensors.min():.4f}, max: {depth_tensors.max():.4f}")

    # 4. Visualize samples
    print("\n4. Creating visualization...")

    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 4))

    # Denormalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(num_samples):
        # RGB
        rgb_img = rgb_samples[i]["pixel_values"]
        rgb_img = (rgb_img * std + mean).clamp(0, 1)
        rgb_label = rgb_samples[i]["labels"]
        axes[0, i].imshow(rgb_img.permute(1, 2, 0).numpy())
        axes[0, i].set_title(f"RGB\nLabel: {rgb_label}", fontsize=8)
        axes[0, i].axis('off')

        # Depth
        depth_img = depth_samples[i]["pixel_values"]
        depth_img = (depth_img * std + mean).clamp(0, 1)
        depth_label = depth_samples[i]["labels"]
        # Show as grayscale (all channels same)
        axes[1, i].imshow(depth_img[0].numpy(), cmap='viridis')
        axes[1, i].set_title(f"Depth\nLabel: {depth_label}", fontsize=8)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel("RGB", fontsize=10)
    axes[1, 0].set_ylabel("Depth", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved to: {save_path}")
    else:
        plt.savefig("depth_dataset_test.png", dpi=150, bbox_inches='tight')
        print(f"   Saved to: depth_dataset_test.png")

    plt.close()

    # 5. Test dataloader
    print("\n5. Testing mixed dataloader...")
    from datasets_with_depth import create_dataloaders_mixed

    try:
        train_loader, val_loader = create_dataloaders_mixed(
            data_root,
            processor,
            train_variant="mixed_depth",
            val_variant="imagenet",
            batch_size=32,
            num_workers=0,  # Use 0 for testing
        )

        # Get one batch
        batch = next(iter(train_loader))
        print(f"   Batch shape: {batch['pixel_values'].shape}")
        print(f"   Labels shape: {batch['labels'].shape}")
        print(f"   Labels range: {batch['labels'].min()} - {batch['labels'].max()}")
        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"   ERROR in dataloader: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Test depth dataset integration")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to data directory")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Number of samples to visualize")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Path to save visualization")

    args = parser.parse_args()
    test_depth_dataset(args.data_root, args.num_samples, args.save_path)


if __name__ == "__main__":
    main()
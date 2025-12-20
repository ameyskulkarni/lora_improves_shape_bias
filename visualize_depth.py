"""Visualize RGB and depth images side by side."""
import argparse
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def visualize_rgb_depth_pairs(
    data_root: str,
    split: str = "val",
    num_samples: int = 5,
    random_seed: int = 42,
    save_path: str = None,
    class_name: str = None,
):
    """Visualize RGB and corresponding depth images side by side.
    
    Args:
        data_root: Path to data directory
        split: "train" or "val"
        num_samples: Number of samples to visualize
        random_seed: Random seed for sampling
        save_path: Optional path to save the visualization
        class_name: Optional specific class to visualize (e.g., 'n01440764')
    """
    rgb_root = Path(data_root) / "imagenet" / split
    depth_root = Path(data_root) / "imagenet_depth" / split
    
    if not rgb_root.exists():
        raise ValueError(f"RGB directory not found: {rgb_root}")
    if not depth_root.exists():
        raise ValueError(f"Depth directory not found: {depth_root}")
    
    # Get class directories
    if class_name:
        class_dirs = [rgb_root / class_name]
        if not class_dirs[0].exists():
            raise ValueError(f"Class directory not found: {class_dirs[0]}")
    else:
        class_dirs = sorted([d for d in rgb_root.iterdir() if d.is_dir()])
    
    # Collect all image pairs
    image_pairs = []
    for class_dir in class_dirs:
        class_name = class_dir.name
        depth_class_dir = depth_root / class_name
        
        if not depth_class_dir.exists():
            continue
        
        # Get RGB images
        rgb_images = (
            list(class_dir.glob("*.JPEG")) +
            list(class_dir.glob("*.jpg")) +
            list(class_dir.glob("*.png"))
        )
        
        for rgb_path in rgb_images:
            # Check for corresponding depth (try both .png and .npy)
            depth_png = depth_class_dir / f"{rgb_path.stem}.png"
            depth_npy = depth_class_dir / f"{rgb_path.stem}.npy"
            
            if depth_png.exists():
                image_pairs.append((rgb_path, depth_png, "png"))
            elif depth_npy.exists():
                image_pairs.append((rgb_path, depth_npy, "npy"))
    
    if len(image_pairs) == 0:
        raise ValueError("No matching RGB-depth pairs found!")
    
    print(f"Found {len(image_pairs)} RGB-depth pairs")
    
    # Sample random pairs
    random.seed(random_seed)
    sampled_pairs = random.sample(image_pairs, min(num_samples, len(image_pairs)))
    
    # Create visualization
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (rgb_path, depth_path, depth_type) in enumerate(sampled_pairs):
        # Load RGB
        rgb = Image.open(rgb_path).convert("RGB")
        
        # Load depth
        if depth_type == "npy":
            depth = np.load(depth_path)
        else:
            depth = np.array(Image.open(depth_path))
        
        # Plot RGB
        axes[idx, 0].imshow(rgb)
        axes[idx, 0].set_title(f"RGB: {rgb_path.parent.name}/{rgb_path.name}", fontsize=10)
        axes[idx, 0].axis('off')
        
        # Plot depth
        im = axes[idx, 1].imshow(depth, cmap='plasma')
        axes[idx, 1].set_title(f"Depth: {depth_path.name}", fontsize=10)
        axes[idx, 1].axis('off')
        
        # Add colorbar for depth
        plt.colorbar(im, ax=axes[idx, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()


def visualize_grid(
    data_root: str,
    split: str = "val",
    num_samples: int = 16,
    random_seed: int = 42,
    save_path: str = None,
):
    """Visualize multiple RGB-depth pairs in a grid."""
    rgb_root = Path(data_root) / "imagenet" / split
    depth_root = Path(data_root) / "imagenet_depth" / split
    
    if not rgb_root.exists():
        raise ValueError(f"RGB directory not found: {rgb_root}")
    if not depth_root.exists():
        raise ValueError(f"Depth directory not found: {depth_root}")
    
    # Collect image pairs
    image_pairs = []
    class_dirs = sorted([d for d in rgb_root.iterdir() if d.is_dir()])
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        depth_class_dir = depth_root / class_name
        
        if not depth_class_dir.exists():
            continue
        
        rgb_images = (
            list(class_dir.glob("*.JPEG")) +
            list(class_dir.glob("*.jpg")) +
            list(class_dir.glob("*.png"))
        )
        
        for rgb_path in rgb_images:
            depth_png = depth_class_dir / f"{rgb_path.stem}.png"
            depth_npy = depth_class_dir / f"{rgb_path.stem}.npy"
            
            if depth_png.exists():
                image_pairs.append((rgb_path, depth_png, "png"))
            elif depth_npy.exists():
                image_pairs.append((rgb_path, depth_npy, "npy"))
    
    print(f"Found {len(image_pairs)} RGB-depth pairs")
    
    # Sample pairs
    random.seed(random_seed)
    sampled_pairs = random.sample(image_pairs, min(num_samples, len(image_pairs)))
    
    # Create grid (4 columns: RGB, Depth, RGB, Depth, ...)
    n_rows = (num_samples + 1) // 2
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    axes = axes.reshape(n_rows, 4)
    
    for idx, (rgb_path, depth_path, depth_type) in enumerate(sampled_pairs):
        row = idx // 2
        col_offset = (idx % 2) * 2
        
        # Load images
        rgb = Image.open(rgb_path).convert("RGB")
        if depth_type == "npy":
            depth = np.load(depth_path)
        else:
            depth = np.array(Image.open(depth_path))
        
        # Plot RGB
        axes[row, col_offset].imshow(rgb)
        axes[row, col_offset].set_title(f"{rgb_path.parent.name}", fontsize=8)
        axes[row, col_offset].axis('off')
        
        # Plot depth
        axes[row, col_offset + 1].imshow(depth, cmap='plasma')
        axes[row, col_offset + 1].set_title("Depth", fontsize=8)
        axes[row, col_offset + 1].axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, n_rows * 2):
        row = idx // 2
        col_offset = (idx % 2) * 2
        axes[row, col_offset].axis('off')
        axes[row, col_offset + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize RGB and depth images")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to data directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save visualization (if not provided, will display)",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default=None,
        help="Specific class to visualize (e.g., n01440764)",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Use grid layout instead of side-by-side",
    )
    
    args = parser.parse_args()
    
    if args.grid:
        visualize_grid(
            data_root=args.data_root,
            split=args.split,
            num_samples=args.num_samples,
            random_seed=args.random_seed,
            save_path=args.save_path,
        )
    else:
        visualize_rgb_depth_pairs(
            data_root=args.data_root,
            split=args.split,
            num_samples=args.num_samples,
            random_seed=args.random_seed,
            save_path=args.save_path,
            class_name=args.class_name,
        )


if __name__ == "__main__":
    main()

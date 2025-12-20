"""Generate depth maps for ImageNet train/val using Depth Anything V2."""
import argparse
import os
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np


def generate_depth_maps(
    data_root: str,
    split: str = "train",
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    device: str = "cuda",
    save_as_numpy: bool = False,
):
    """Generate depth maps for all ImageNet images.
    
    Args:
        data_root: Path to data directory (should contain imagenet/train or imagenet/val)
        split: "train" or "val"
        model_name: HuggingFace model name for Depth Anything V2
        device: "cuda" or "cpu"
        save_as_numpy: If True, save as .npy files, else save as PNG
    """
    # Setup paths
    rgb_root = Path(data_root) / "imagenet" / split
    depth_root = Path(data_root) / "imagenet_depth" / split
    
    if not rgb_root.exists():
        raise ValueError(f"RGB directory not found: {rgb_root}")
    
    print(f"RGB root: {rgb_root}")
    print(f"Depth root: {depth_root}")
    
    # Load Depth Anything V2 model
    print(f"\nLoading {model_name}...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Get all class directories
    class_dirs = sorted([d for d in rgb_root.iterdir() if d.is_dir()])
    print(f"\nFound {len(class_dirs)} classes")
    
    total_images = 0
    for class_dir in class_dirs:
        images = list(class_dir.glob("*.JPEG")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        total_images += len(images)
    print(f"Total images to process: {total_images}")
    
    # Process each class
    processed = 0
    with torch.no_grad():
        for class_dir in tqdm(class_dirs, desc="Classes"):
            class_name = class_dir.name
            
            # Create corresponding depth directory
            depth_class_dir = depth_root / class_name
            depth_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all images in this class
            image_files = (
                list(class_dir.glob("*.JPEG")) +
                list(class_dir.glob("*.jpg")) +
                list(class_dir.glob("*.png"))
            )
            
            for img_path in tqdm(image_files, desc=f"{class_name}", leave=False):
                # Skip if already processed
                if save_as_numpy:
                    depth_path = depth_class_dir / f"{img_path.stem}.npy"
                else:
                    depth_path = depth_class_dir / f"{img_path.stem}.png"
                
                if depth_path.exists():
                    processed += 1
                    continue
                
                try:
                    # Load RGB image
                    image = Image.open(img_path).convert("RGB")
                    
                    # Process with Depth Anything V2
                    inputs = processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    outputs = model(**inputs)
                    predicted_depth = outputs.predicted_depth
                    
                    # Interpolate to original image size
                    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=image.size[::-1],  # (height, width)
                        mode="bicubic",
                        align_corners=False,
                    )
                    
                    # Convert to numpy
                    depth = prediction.squeeze().cpu().numpy()
                    
                    # Save depth map
                    if save_as_numpy:
                        np.save(depth_path, depth)
                    else:
                        # Normalize to 0-255 for PNG
                        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
                        Image.fromarray(depth_uint8).save(depth_path)
                    
                    processed += 1
                    
                except Exception as e:
                    print(f"\nError processing {img_path}: {e}")
                    continue
    
    print(f"\n✓ Completed! Processed {processed}/{total_images} images")
    print(f"Depth maps saved to: {depth_root}")


def main():
    parser = argparse.ArgumentParser(description="Generate depth maps for ImageNet")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to data directory (should contain imagenet/train or imagenet/val)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="HuggingFace model name (options: Depth-Anything-V2-Small/Base/Large-hf)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--save-as-numpy",
        action="store_true",
        help="Save as .npy files instead of PNG (preserves float precision)",
    )
    
    args = parser.parse_args()
    
    generate_depth_maps(
        data_root=args.data_root,
        split=args.split,
        model_name=args.model_name,
        device=args.device,
        save_as_numpy=args.save_as_numpy,
    )


if __name__ == "__main__":
    main()

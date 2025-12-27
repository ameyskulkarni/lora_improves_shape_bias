"""
Download ResNet checkpoints from the official texture-vs-shape repository.

This script downloads the official pretrained ResNet-50 models used in
"ImageNet-trained CNNs are biased towards texture" (Geirhos et al., ICLR 2019)
"""

import os
import urllib.request
import argparse
from pathlib import Path


# Official checkpoint URLs from https://github.com/rgeirhos/texture-vs-shape
CHECKPOINTS = {
    "resnet50_sin": {
        "filename": "resnet50_train_60_epochs-c8e5653e.pth.tar",
        "url": "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar",
        "description": "ResNet-50 trained on Stylized-ImageNet (SIN) for 60 epochs",
        "expected_shape_bias": "~81%"
    },
    "resnet50_sin_in": {
        "filename": "resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar",
        "url": "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar",
        "description": "ResNet-50 trained jointly on SIN+IN for 45 epochs",
        "expected_shape_bias": "~50-60% (between IN and SIN)"
    },
    "shape_resnet": {
        "filename": "resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar",
        "url": "https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar",
        "description": "Shape-ResNet: SIN+IN model fine-tuned on IN for 60 epochs",
        "expected_shape_bias": "~70-75% (higher than SIN+IN)"
    }
}


def download_checkpoint(checkpoint_key: str, output_dir: str):
    """Download a checkpoint if it doesn't exist."""
    
    if checkpoint_key not in CHECKPOINTS:
        print(f"Unknown checkpoint: {checkpoint_key}")
        print(f"Available checkpoints: {list(CHECKPOINTS.keys())}")
        return False
    
    info = CHECKPOINTS[checkpoint_key]
    output_path = os.path.join(output_dir, info["filename"])
    
    if os.path.exists(output_path):
        print(f"✓ Checkpoint already exists: {output_path}")
        return True
    
    print(f"\nDownloading {checkpoint_key}...")
    print(f"  Description: {info['description']}")
    print(f"  Expected shape bias: {info['expected_shape_bias']}")
    print(f"  URL: {info['url']}")
    print(f"  Output: {output_path}")
    
    try:
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
        
        urllib.request.urlretrieve(info['url'], output_path, reporthook=report_progress)
        print(f"\n✓ Downloaded successfully: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading {checkpoint_key}: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download official ResNet checkpoints from texture-vs-shape repository"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints/resnet_sin",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        choices=list(CHECKPOINTS.keys()) + ["all"],
        default="all",
        help="Which checkpoint to download"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("ResNet Checkpoint Downloader")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Download specified checkpoints
    if args.checkpoint == "all":
        checkpoints_to_download = list(CHECKPOINTS.keys())
    else:
        checkpoints_to_download = [args.checkpoint]
    
    success_count = 0
    for checkpoint_key in checkpoints_to_download:
        if download_checkpoint(checkpoint_key, args.output_dir):
            success_count += 1
        print()
    
    # Summary
    print("="*80)
    print(f"Downloaded {success_count}/{len(checkpoints_to_download)} checkpoints successfully")
    print("="*80)
    
    if success_count > 0:
        print(f"\nNext steps:")
        print(f"1. Run the validation test:")
        print(f"   python test_shape_bias_validation.py \\")
        print(f"       --data-root /path/to/your/data \\")
        print(f"       --checkpoint-dir {args.output_dir} \\")
        print(f"       --test resnet")
        print()
        print(f"Expected results:")
        print(f"  - ResNet-50 (ImageNet from torchvision): ~22% shape bias")
        print(f"  - ResNet-50 (SIN): ~81% shape bias")
        print(f"  - ResNet-50 (SIN+IN): ~50-60% shape bias")
        print(f"  - Shape-ResNet: ~70-75% shape bias")


if __name__ == "__main__":
    main()

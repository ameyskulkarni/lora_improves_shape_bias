"""
Quick diagnostic for DeiT preprocessing and accuracy issues.

This script helps identify the exact cause of DeiT shape bias discrepancies.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import argparse


def test_deit_accuracy_with_different_preprocessing(
    data_root: str,
    model_name: str = "facebook/deit-tiny-patch16-224"
):
    """
    Test DeiT accuracy with different preprocessing to identify the issue.
    
    Expected DeiT-Tiny accuracy on ImageNet: ~72.2%
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print(f"DeiT Preprocessing Diagnostic")
    print("="*80)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print()
    
    # Load model
    print("Loading model...")
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = model.to(device).eval()
    
    # Load ImageNet validation set
    imagenet_val = Path(data_root) / "imagenet" / "val"
    if not imagenet_val.exists():
        print(f"ERROR: ImageNet validation set not found at {imagenet_val}")
        return
    
    print(f"Loading ImageNet val from: {imagenet_val}")
    
    # Test configurations
    configs = {
        "HF Default (WRONG)": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "interpolation": transforms.InterpolationMode.BILINEAR,
            "resize": 256,
            "crop": 224,
        },
        "ImageNet (CORRECT)": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": transforms.InterpolationMode.BICUBIC,
            "resize": 256,
            "crop": 224,
        },
        "ImageNet + BILINEAR": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": transforms.InterpolationMode.BILINEAR,
            "resize": 256,
            "crop": 224,
        },
        "ImageNet + resize=224": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "interpolation": transforms.InterpolationMode.BICUBIC,
            "resize": 224,  # No resize step, direct crop
            "crop": 224,
        },
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print("\n" + "-"*80)
        print(f"Testing: {config_name}")
        print("-"*80)
        print(f"  mean:          {config['mean']}")
        print(f"  std:           {config['std']}")
        print(f"  interpolation: {config['interpolation']}")
        print(f"  resize:        {config['resize']}")
        print(f"  crop:          {config['crop']}")
        
        # Build transform
        transform_ops = []
        if config['resize'] != config['crop']:
            transform_ops.append(
                transforms.Resize(config['resize'], interpolation=config['interpolation'])
            )
        transform_ops.extend([
            transforms.CenterCrop(config['crop']),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['mean'], std=config['std'])
        ])
        
        transform = transforms.Compose(transform_ops)
        
        # Create dataset and loader
        dataset = ImageFolder(str(imagenet_val), transform=transform)
        loader = DataLoader(
            dataset, 
            batch_size=128,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Evaluate
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Evaluating", leave=False):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                preds = outputs.logits.argmax(-1)
                
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        results[config_name] = accuracy
        
        print(f"\n  Accuracy: {accuracy*100:.2f}%")
        
        # Compare to expected
        expected_deit_tiny = 0.722  # 72.2%
        diff = (accuracy - expected_deit_tiny) * 100
        
        if abs(diff) < 1.0:
            status = "✓ MATCHES EXPECTED"
        elif diff < -3.0:
            status = "✗ SIGNIFICANTLY LOWER"
        else:
            status = "~ CLOSE"
        
        print(f"  vs Expected (72.2%): {diff:+.2f}% - {status}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Configuration':<30} {'Accuracy':<15} {'vs Expected':<15}")
    print("-"*80)
    
    expected = 0.722
    for config_name, accuracy in results.items():
        diff = (accuracy - expected) * 100
        print(f"{config_name:<30} {accuracy*100:>6.2f}%        {diff:>+6.2f}%")
    
    print("="*80)
    
    # Diagnosis
    print("\nDIAGNOSIS:")
    best_config = max(results.items(), key=lambda x: x[1])
    print(f"  Best configuration: {best_config[0]} ({best_config[1]*100:.2f}%)")
    
    if best_config[1] >= 0.71:  # Within 1% of expected
        print(f"  ✓ Accuracy is good! Your preprocessing is likely correct.")
    else:
        print(f"  ✗ All configurations underperform. Possible issues:")
        print(f"    - Wrong model weights")
        print(f"    - Dataset corruption")
        print(f"    - Other preprocessing issue")
    
    # Check which factor matters most
    wrong_norm = results["HF Default (WRONG)"]
    correct_norm = results["ImageNet (CORRECT)"]
    norm_diff = (correct_norm - wrong_norm) * 100
    
    print(f"\n  Impact of correct normalization: {norm_diff:+.2f}%")
    if abs(norm_diff) > 2.0:
        print(f"  ⚠ CRITICAL: Normalization makes a big difference!")
        print(f"    Always use ImageNet normalization, NOT [0.5, 0.5, 0.5]")


def compare_model_sources(data_root: str):
    """Compare models loaded from torch.hub vs HuggingFace."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "="*80)
    print("Comparing DeiT-Tiny from different sources")
    print("="*80)
    
    # Load ImageNet validation set
    imagenet_val = Path(data_root) / "imagenet" / "val"
    if not imagenet_val.exists():
        print(f"ERROR: ImageNet validation set not found at {imagenet_val}")
        return
    
    # Correct preprocessing
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolder(str(imagenet_val), transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    results = {}
    
    # Test 1: HuggingFace
    print("\n1. Testing HuggingFace model...")
    try:
        model = ViTForImageClassification.from_pretrained("facebook/deit-tiny-patch16-224")
        model = model.to(device).eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="HuggingFace"):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = outputs.logits.argmax(-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        results["HuggingFace"] = correct / total
        print(f"   Accuracy: {results['HuggingFace']*100:.2f}%")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: torch.hub
    print("\n2. Testing torch.hub model...")
    try:
        model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        model = model.to(device).eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="torch.hub"):
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                preds = logits.argmax(-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        results["torch.hub"] = correct / total
        print(f"   Accuracy: {results['torch.hub']*100:.2f}%")
    except Exception as e:
        print(f"   Error: {e}")
        print(f"   Note: torch.hub requires internet connection on first run")
    
    # Comparison
    if len(results) == 2:
        diff = (results["HuggingFace"] - results["torch.hub"]) * 100
        print(f"\n" + "="*80)
        print(f"Difference (HuggingFace - torch.hub): {diff:+.2f}%")
        
        if abs(diff) < 0.5:
            print(f"✓ Models are essentially identical")
        elif abs(diff) < 2.0:
            print(f"~ Small difference, likely due to minor implementation details")
        else:
            print(f"✗ Significant difference - models may be different!")
        print("="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--model", type=str, default="facebook/deit-tiny-patch16-224")
    parser.add_argument("--compare-sources", action="store_true",
                       help="Also compare torch.hub vs HuggingFace")
    
    args = parser.parse_args()
    
    # Test preprocessing
    test_deit_accuracy_with_different_preprocessing(args.data_root, args.model)
    
    # Compare sources if requested
    if args.compare_sources:
        compare_model_sources(args.data_root)


if __name__ == "__main__":
    main()

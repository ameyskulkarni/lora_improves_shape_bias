"""
Test script to validate cue-conflict shape bias calculations.

This script tests:
1. ResNet-50 models (IN, SIN, SIN+IN) against paper's reported values
2. Different DeiT loading methods (torch.hub vs HuggingFace/timm)

Expected Results from Geirhos et al. (ICLR 2019):
- ResNet-50 (IN): ~21-22% shape bias
- ResNet-50 (SIN): ~81% shape bias  
- ResNet-50 (SIN+IN): Higher than IN, lower than SIN
- DeiT-Tiny: Unknown (not in original paper)
"""

import sys
import os
import traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import from project directory if not in current directory
try:
    from cue_conflict_evaluator import CueConflictEvaluator
except ImportError:
    # Try adding project directory to path
    project_dir = "/mnt/project"
    if os.path.exists(project_dir):
        sys.path.insert(0, project_dir)
        from cue_conflict_evaluator import CueConflictEvaluator
    else:
        raise ImportError(
            "Could not find cue_conflict_evaluator.py\n"
            "Please run this script from your project directory or ensure "
            "cue_conflict_evaluator.py is in the same directory."
        )

import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from pathlib import Path
import argparse
from typing import Dict, Tuple


# =============================================================================
# Custom Evaluator for ResNet (matching original paper preprocessing)
# =============================================================================

class ResNetCueConflictEvaluator(CueConflictEvaluator):
    """
    Modified evaluator that uses ResNet-style preprocessing instead of ViT-style.

    Differences from default:
    - Uses BILINEAR interpolation (ResNet default) instead of BICUBIC (ViT default)
    - Uses simple Resize(256) instead of eval_resize = size/0.875
    """

    def _build_transform(self):
        """Build ResNet-style transforms matching the original paper."""
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# =============================================================================
# Model Loading Functions
# =============================================================================

def load_resnet50_imagenet(device='cuda'):
    """Load pretrained ResNet-50 from torchvision (trained on ImageNet)."""
    print("\n" + "="*70)
    print("Loading ResNet-50 (ImageNet) from torchvision")
    print("="*70)

    model = models.resnet50(pretrained=True)
    model = model.to(device).eval()

    # Standard ImageNet preprocessing
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    class ResNetWrapper(nn.Module):
        """Wrapper to match evaluator interface."""
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            logits = self.model(pixel_values)
            return type('obj', (object,), {'logits': logits})()

    # Create a simple processor-like object
    class SimpleProcessor:
        def __init__(self):
            self.size = 224
            self.image_mean = [0.485, 0.456, 0.406]
            self.image_std = [0.229, 0.224, 0.225]

    processor = SimpleProcessor()
    wrapped_model = ResNetWrapper(model)

    return wrapped_model, processor, model


def load_resnet50_sin(checkpoint_path: str, device='cuda'):
    """
    Load ResNet-50 trained on Stylized-ImageNet (SIN).

    Download checkpoint from:
    https://github.com/rgeirhos/texture-vs-shape

    The authors provide models at:
    - ResNet-50 trained on SIN: resnet50_train_60_epochs-c8e5653e.pth.tar
    - ResNet-50 trained on SIN+IN: resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar
    - Shape-ResNet (SIN+IN→IN): resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar
    """
    print("\n" + "="*70)
    print(f"Loading ResNet-50 (SIN) from checkpoint: {checkpoint_path}")
    print("="*70)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Download from: https://github.com/rgeirhos/texture-vs-shape/tree/master/models"
        )

    model = models.resnet50(pretrained=False)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present (from DataParallel)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # Same preprocessing as ImageNet ResNet
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    class ResNetWrapper(nn.Module):
        """Wrapper to match evaluator interface."""
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            logits = self.model(pixel_values)
            return type('obj', (object,), {'logits': logits})()

    class SimpleProcessor:
        def __init__(self):
            self.size = 224
            self.image_mean = [0.485, 0.456, 0.406]
            self.image_std = [0.229, 0.224, 0.225]

    processor = SimpleProcessor()
    wrapped_model = ResNetWrapper(model)

    return wrapped_model, processor, model


def load_deit_huggingface(model_name="facebook/deit-tiny-patch16-224", device='cuda'):
    """Load DeiT from HuggingFace (current method)."""
    print("\n" + "="*70)
    print(f"Loading {model_name} from HuggingFace")
    print("="*70)

    model = ViTForImageClassification.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = model.to(device).eval()

    print(f"HuggingFace processor normalization:")
    print(f"  mean: {processor.image_mean}")
    print(f"  std: {processor.image_std}")
    print(f"  NOTE: This is INCORRECT for DeiT! Should use ImageNet values.")

    return model, processor


def load_deit_torchhub(model_name="deit_tiny_patch16_224", device='cuda'):
    """Load DeiT from torch.hub (original Facebook method)."""
    print("\n" + "="*70)
    print(f"Loading {model_name} from torch.hub (Facebook)")
    print("="*70)

    # Load from Facebook's repository
    model = torch.hub.load('facebookresearch/deit:main', model_name, pretrained=True)
    model = model.to(device).eval()

    # Create a compatible processor with CORRECT normalization
    class TorchHubProcessor:
        def __init__(self):
            self.size = 224
            self.image_mean = [0.485, 0.456, 0.406]  # Correct ImageNet normalization
            self.image_std = [0.229, 0.224, 0.225]

    processor = TorchHubProcessor()

    # Wrap model to match HuggingFace interface
    class DeiTWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            logits = self.model(pixel_values)
            return type('obj', (object,), {'logits': logits})()

    wrapped_model = DeiTWrapper(model)

    return wrapped_model, processor


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_shape_bias(
    model,
    processor,
    data_root: str,
    device: str = 'cuda',
    model_name: str = "Model",
    use_resnet_preprocessing: bool = False,
) -> Dict[str, float]:
    """Evaluate shape bias using cue-conflict dataset."""

    cue_conflict_path = os.path.join(data_root, "cue-conflict")

    if not os.path.exists(cue_conflict_path):
        raise FileNotFoundError(
            f"Cue-conflict dataset not found at {cue_conflict_path}\n"
            "Download from: https://github.com/rgeirhos/texture-vs-shape/tree/master/stimuli/style-transfer-preprocessed-512"
        )

    # Choose evaluator based on model type
    EvaluatorClass = ResNetCueConflictEvaluator if use_resnet_preprocessing else CueConflictEvaluator

    evaluator = EvaluatorClass(
        model=model,
        processor=processor,
        cue_conflict_path=cue_conflict_path,
        device=device,
        batch_size=32,
        num_workers=4,
        aggregation="mean",  # Recommended by paper
    )

    print(f"\nEvaluating {model_name}...")
    if use_resnet_preprocessing:
        print("  Using ResNet-style preprocessing (BILINEAR, Resize(256))")
    else:
        print("  Using ViT-style preprocessing (BICUBIC, Resize(256) via 0.875 crop)")

    results = evaluator.evaluate(verbose=True)

    return results


def print_comparison_table(results_dict: Dict[str, Dict], expected_values: Dict = None):
    """Print a nice comparison table of results."""

    print("\n" + "="*90)
    print("SHAPE BIAS COMPARISON")
    print("="*90)
    print(f"{'Model':<40} {'Shape Bias':<15} {'Expected':<15} {'Difference':<15}")
    print("-"*90)

    for model_name, results in results_dict.items():
        shape_bias = results['shape_bias']
        expected = expected_values.get(model_name, "N/A") if expected_values else "N/A"

        if expected != "N/A":
            diff = f"{(shape_bias - expected)*100:+.2f}%"
        else:
            diff = "N/A"

        print(f"{model_name:<40} {shape_bias:.2%}      {expected if isinstance(expected, str) else f'{expected:.2%}':<15} {diff:<15}")

    print("="*90)

    # Detailed breakdown
    print("\nDETAILED BREAKDOWN:")
    print("-"*90)
    for model_name, results in results_dict.items():
        print(f"\n{model_name}:")
        print(f"  Shape correct:   {results['shape_correct']:>5}")
        print(f"  Texture correct: {results['texture_correct']:>5}")
        print(f"  Both correct:    {results['both_correct']:>5}")
        print(f"  Neither correct: {results['neither_correct']:>5}")
        print(f"  Total evaluated: {results['total']:>5}")
        print(f"  Shape bias:      {results['shape_bias']:.2%}")


# =============================================================================
# Main Test Functions
# =============================================================================

def test_resnet_models(data_root: str, checkpoint_dir: str = None):
    """
    Test ResNet-50 models to validate shape bias calculations.

    Args:
        data_root: Root directory containing cue-conflict dataset
        checkpoint_dir: Directory containing downloaded ResNet SIN checkpoints
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    # Expected values from the paper (Geirhos et al., ICLR 2019)
    expected = {
        "ResNet-50 (ImageNet)": 0.22,  # ~22% from Figure 4
        "ResNet-50 (SIN)": 0.81,        # ~81% from Figure 5
    }

    # Test 1: ResNet-50 trained on ImageNet (torchvision)
    try:
        model, processor, _ = load_resnet50_imagenet(device)
        results["ResNet-50 (ImageNet)"] = evaluate_shape_bias(
            model, processor, data_root, device, "ResNet-50 (ImageNet)",
            use_resnet_preprocessing=True
        )
    except Exception as e:
        print(f"Error testing ResNet-50 (ImageNet): {e}")
        import traceback
        traceback.print_exc()

    # Test 2: ResNet-50 trained on SIN (if checkpoint provided)
    if checkpoint_dir:
        checkpoint_files = {
            "ResNet-50 (SIN)": "resnet50_train_60_epochs-c8e5653e.pth.tar",
            "ResNet-50 (SIN+IN)": "resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar",
            "Shape-ResNet (SIN+IN→IN)": "resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar",
        }

        for model_name, filename in checkpoint_files.items():
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            if os.path.exists(checkpoint_path):
                try:
                    model, processor, _ = load_resnet50_sin(checkpoint_path, device)
                    results[model_name] = evaluate_shape_bias(
                        model, processor, data_root, device, model_name,
                        use_resnet_preprocessing=True
                    )
                except Exception as e:
                    print(f"Error testing {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"\nSkipping {model_name} - checkpoint not found at {checkpoint_path}")
    else:
        print("\nSkipping SIN/SIN+IN models - no checkpoint directory provided")
        print("Download checkpoints from: https://github.com/rgeirhos/texture-vs-shape/tree/master/models")

    # Print comparison
    print_comparison_table(results, expected)

    return results


def test_deit_loading_methods(data_root: str, model_name: str = "deit_tiny_patch16_224"):
    """
    Compare DeiT loaded from torch.hub vs HuggingFace.

    This tests whether the discrepancy is due to different model sources.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}

    # Map model names
    hub_name = model_name
    hf_name = f"facebook/{model_name.replace('_', '-')}"

    # Test 1: Load from torch.hub (original Facebook)
    print("\n" + "="*70)
    print("TEST 1: Loading from torch.hub (Facebook original)")
    print("="*70)
    try:
        model, processor = load_deit_torchhub(hub_name, device)
        results["DeiT (torch.hub)"] = evaluate_shape_bias(
            model, processor, data_root, device, "DeiT (torch.hub)"
        )
    except Exception as e:
        print(f"Error loading from torch.hub: {e}")
        print("Note: torch.hub requires internet connection on first run")

    # Test 2: Load from HuggingFace (current method)
    print("\n" + "="*70)
    print("TEST 2: Loading from HuggingFace/timm")
    print("="*70)
    try:
        model, processor = load_deit_huggingface(hf_name, device)

        # Test with INCORRECT normalization (HF default)
        print("\n>>> Testing with INCORRECT HuggingFace normalization [0.5, 0.5, 0.5]")
        results["DeiT (HF - wrong norm)"] = evaluate_shape_bias(
            model, processor, data_root, device, "DeiT (HF - wrong norm)"
        )

        # Test with CORRECT normalization (ImageNet)
        print("\n>>> Testing with CORRECT ImageNet normalization")
        processor.image_mean = [0.485, 0.456, 0.406]
        processor.image_std = [0.229, 0.224, 0.225]
        results["DeiT (HF - correct norm)"] = evaluate_shape_bias(
            model, processor, data_root, device, "DeiT (HF - correct norm)"
        )

    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")

    # Print comparison
    print_comparison_table(results)

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate shape bias calculations against known ResNet results"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing cue-conflict dataset"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["resnet", "deit", "both"],
        default="both",
        help="Which test to run"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing ResNet SIN/SIN+IN checkpoints"
    )
    parser.add_argument(
        "--deit-model",
        type=str,
        default="deit_tiny_patch16_224",
        help="DeiT model name (e.g., deit_tiny_patch16_224, deit_small_patch16_224)"
    )

    args = parser.parse_args()

    print("\n" + "="*90)
    print("SHAPE BIAS VALIDATION TEST")
    print("="*90)
    print(f"Data root: {args.data_root}")
    print(f"Test type: {args.test}")
    print("="*90)

    # Run tests
    if args.test in ["resnet", "both"]:
        print("\n" + "#"*90)
        print("# TEST 1: RESNET MODELS (Validation against paper)")
        print("#"*90)
        test_resnet_models(args.data_root, args.checkpoint_dir)

    if args.test in ["deit", "both"]:
        print("\n" + "#"*90)
        print("# TEST 2: DEIT LOADING METHODS (torch.hub vs HuggingFace)")
        print("#"*90)
        test_deit_loading_methods(args.data_root, args.deit_model)

    print("\n" + "="*90)
    print("VALIDATION COMPLETE")
    print("="*90)


if __name__ == "__main__":
    main()
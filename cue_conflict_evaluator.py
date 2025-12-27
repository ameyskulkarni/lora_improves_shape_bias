"""
cue_conflict_evaluator.py - Cue-Conflict Shape Bias Evaluation

This module implements the shape bias metric from:
"ImageNet-trained CNNs are biased towards texture; increasing shape bias 
improves accuracy and robustness" (Geirhos et al., ICLR 2019)

The calculation follows the official methodology:
1. Evaluate on cue-conflict stimuli (1,280 images)
2. Map ImageNet 1000-class predictions to 16 entry-level categories (using average)
3. Exclude non-conflict images (where shape == texture)
4. Keep only "correct" predictions (predicted either shape OR texture category)
5. Shape Bias = shape_correct / (shape_correct + texture_correct)

Reference: https://github.com/rgeirhos/texture-vs-shape
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# =============================================================================
# 16 Entry-Level Categories used in the cue-conflict dataset
# =============================================================================
SIXTEEN_CLASSES = sorted([
    "airplane", "bear", "bicycle", "bird", "boat", "bottle", 
    "car", "cat", "chair", "clock", "dog", "elephant", 
    "keyboard", "knife", "oven", "truck"
])

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SIXTEEN_CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(SIXTEEN_CLASSES)}




# =============================================================================
# ImageNet 1000 → 16 Class Mapping
#
# EXACT mapping from the OFFICIAL model-vs-human / texture-vs-shape repos:
# Source: helper/human_categories.py
# =============================================================================

IMAGENET_TO_16CLASS_MAPPING = {
    "airplane" : [404],
    "bear" : [294, 295, 296, 297],
    "bicycle" : [444, 671],
    "bird" : [8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23,
                    24, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92, 93,
                    94, 95, 96, 98, 99, 100, 127, 128, 129, 130, 131,
                    132, 133, 135, 136, 137, 138, 139, 140, 141, 142,
                    143, 144, 145],
    "boat" : [472, 554, 625, 814, 914],
    "bottle" : [440, 720, 737, 898, 899, 901, 907],
    "car" : [436, 511, 817],
    "cat" : [281, 282, 283, 284, 285, 286],
    "chair" : [423, 559, 765, 857],
    "clock" : [409, 530, 892],
    "dog" : [152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
                   162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
                   172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
                   182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
                   193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
                   203, 205, 206, 207, 208, 209, 210, 211, 212, 213,
                   214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
                   224, 225, 226, 228, 229, 230, 231, 232, 233, 234,
                   235, 236, 237, 238, 239, 240, 241, 243, 244, 245,
                   246, 247, 248, 249, 250, 252, 253, 254, 255, 256,
                   257, 259, 261, 262, 263, 265, 266, 267, 268],
    "elephant" : [385, 386],
    "keyboard" : [508, 878],
    "knife" : [499],
    "oven" : [766],
    "truck" : [555, 569, 656, 675, 717, 734, 864, 867],
}


def get_imagenet_to_16class_mapping() -> Dict[int, str]:
    """
    Create a mapping from ImageNet class indices to 16-class categories.
    
    Returns:
        Dictionary mapping ImageNet class index (0-999) to 16-class name
    """
    mapping = {}
    for class_name, imagenet_indices in IMAGENET_TO_16CLASS_MAPPING.items():
        for idx in imagenet_indices:
            mapping[idx] = class_name
    return mapping

def logits_to_16class(logits):
    class_logits = np.zeros(16)
    for i, cls in enumerate(SIXTEEN_CLASSES):
        class_logits[i] = np.mean(logits[IMAGENET_TO_16CLASS_MAPPING[cls]])
    return IDX_TO_CLASS[np.argmax(class_logits)]

def probabilities_to_16class(
    probabilities: np.ndarray,
    aggregation: str = "mean"
) -> Tuple[str, np.ndarray]:
    """
    Convert ImageNet 1000-class probabilities to 16-class prediction.
    
    Uses the AVERAGE aggregation method as recommended in the paper:
    "We here used the average: ImageNet class probabilities were mapped to the 
    corresponding 16-class-ImageNet category using the average of all 
    corresponding fine-grained category probabilities."
    
    Args:
        probabilities: (1000,) array of ImageNet class probabilities
        aggregation: "mean" (recommended) or "sum" or "max"
        
    Returns:
        Tuple of (predicted_class_name, 16-class_probabilities)
    """
    class_probs = np.zeros(16)
    
    for class_idx, class_name in enumerate(SIXTEEN_CLASSES):
        imagenet_indices = IMAGENET_TO_16CLASS_MAPPING[class_name]
        relevant_probs = probabilities[imagenet_indices]
        
        if aggregation == "mean":
            class_probs[class_idx] = np.mean(relevant_probs)
        elif aggregation == "sum":
            class_probs[class_idx] = np.sum(relevant_probs)
        elif aggregation == "max":
            class_probs[class_idx] = np.max(relevant_probs)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    predicted_idx = np.argmax(class_probs)
    predicted_class = IDX_TO_CLASS[predicted_idx]
    
    return predicted_class, class_probs


# =============================================================================
# Cue-Conflict Dataset
# =============================================================================
@dataclass
class CueConflictSample:
    """A single cue-conflict image with shape and texture labels."""
    image_path: str
    shape_class: str
    texture_class: str
    is_conflict: bool  # True if shape != texture
    
    @property
    def shape_idx(self) -> int:
        return CLASS_TO_IDX[self.shape_class]
    
    @property
    def texture_idx(self) -> int:
        return CLASS_TO_IDX[self.texture_class]


class CueConflictDataset(Dataset):
    """
    Dataset for cue-conflict stimuli.
    
    Expected directory structure:
        cue-conflict/
        ├── airplane/
        │   ├── airplane1-bear2.png  (shape=airplane, texture=bear)
        │   ├── airplane1-bicycle3.png
        │   └── ...
        ├── bear/
        │   └── ...
        └── ...
    
    The folder name indicates the SHAPE category.
    The filename format is: {shape}{num}-{texture}{num}.png
    """
    
    def __init__(
        self, 
        root: str, 
        transform: Optional[transforms.Compose] = None,
        include_non_conflict: bool = False
    ):
        """
        Args:
            root: Path to cue-conflict directory
            transform: Image transforms to apply
            include_non_conflict: If True, include images where shape==texture
        """
        self.root = Path(root)
        self.transform = transform
        self.include_non_conflict = include_non_conflict
        self.samples: List[CueConflictSample] = []
        
        self._load_samples()
        
        print(f"CueConflictDataset: {len(self.samples)} samples loaded")
        if not include_non_conflict:
            print("  (non-conflict images excluded)")
    
    def _parse_filename(self, filename: str) -> Tuple[str, str]:
        """
        Parse shape and texture from filename.
        
        Example: "airplane10-bear3.png" -> ("airplane", "bear")
        """
        # Remove extension
        name = Path(filename).stem
        
        # Split by hyphen
        parts = name.split("-")
        if len(parts) != 2:
            raise ValueError(f"Unexpected filename format: {filename}")
        
        # Extract class names (remove trailing numbers)
        shape_part = parts[0]
        texture_part = parts[1]
        
        # Use regex to extract class name (letters only)
        shape_match = re.match(r'^([a-zA-Z]+)\d*$', shape_part)
        texture_match = re.match(r'^([a-zA-Z]+)\d*$', texture_part)
        
        if not shape_match or not texture_match:
            raise ValueError(f"Could not parse: {filename}")
        
        shape_class = shape_match.group(1).lower()
        texture_class = texture_match.group(1).lower()
        
        return shape_class, texture_class
    
    def _load_samples(self):
        """Load all samples from the dataset directory."""
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            
            folder_class = class_dir.name.lower()
            
            if folder_class not in CLASS_TO_IDX:
                print(f"Warning: Skipping unknown class folder: {folder_class}")
                continue
            
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                    continue
                
                try:
                    shape_class, texture_class = self._parse_filename(img_path.name)
                except ValueError as e:
                    print(f"Warning: {e}")
                    continue
                
                # Verify shape matches folder (sanity check)
                if shape_class != folder_class:
                    print(f"Warning: Folder/filename mismatch: {img_path}")
                    continue
                
                is_conflict = shape_class != texture_class
                
                if not self.include_non_conflict and not is_conflict:
                    continue
                
                self.samples.append(CueConflictSample(
                    image_path=str(img_path),
                    shape_class=shape_class,
                    texture_class=texture_class,
                    is_conflict=is_conflict,
                ))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        image = Image.open(sample.image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return {
            "pixel_values": image,
            "shape_label": sample.shape_idx,
            "texture_label": sample.texture_idx,
            "shape_class": sample.shape_class,
            "texture_class": sample.texture_class,
            "is_conflict": sample.is_conflict,
            "image_path": sample.image_path,
        }


# =============================================================================
# Shape Bias Calculator
# =============================================================================
class ShapeBiasCalculator:
    """
    Calculate shape bias from cue-conflict evaluation results.
    
    Shape bias is defined as:
        shape_bias = shape_correct / (shape_correct + texture_correct)
    
    Where we only consider images where the model predicted EITHER 
    the shape OR the texture category (not neither, not both).
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all counters."""
        self.shape_correct = 0
        self.texture_correct = 0
        self.both_correct = 0
        self.neither_correct = 0
        self.total = 0
        
        # Per-class statistics
        self.per_class_shape = {c: 0 for c in SIXTEEN_CLASSES}
        self.per_class_texture = {c: 0 for c in SIXTEEN_CLASSES}
        self.per_class_total = {c: 0 for c in SIXTEEN_CLASSES}
    
    def update(
        self,
        predicted_class: str,
        shape_class: str,
        texture_class: str,
    ):
        """
        Update counters with a single prediction.
        
        Args:
            predicted_class: The 16-class prediction from the model
            shape_class: Ground truth shape category
            texture_class: Ground truth texture category
        """
        self.total += 1
        self.per_class_total[shape_class] += 1
        
        shape_match = predicted_class == shape_class
        texture_match = predicted_class == texture_class
        
        if shape_match and texture_match:
            # This shouldn't happen for conflict images
            print(f"Warning: Shape and Texture classes match")
            self.both_correct += 1
        elif shape_match:
            self.shape_correct += 1
            self.per_class_shape[shape_class] += 1
        elif texture_match:
            self.texture_correct += 1
            self.per_class_texture[texture_class] += 1
        else:
            self.neither_correct += 1
    
    def compute_shape_bias(self) -> float:
        """
        Compute overall shape bias.
        
        Returns:
            Shape bias as a fraction between 0 and 1
            (0 = pure texture bias, 1 = pure shape bias)
        """
        denominator = self.shape_correct + self.texture_correct
        if denominator == 0:
            return 0.0
        return self.shape_correct / denominator
    
    def compute_per_class_shape_bias(self) -> Dict[str, float]:
        """Compute shape bias per shape category."""
        per_class_bias = {}
        for cls in SIXTEEN_CLASSES:
            shape_c = self.per_class_shape[cls]
            texture_c = self.per_class_texture[cls]
            if shape_c + texture_c > 0:
                per_class_bias[cls] = shape_c / (shape_c + texture_c)
            else:
                per_class_bias[cls] = 0.0
        return per_class_bias
    
    def get_summary(self) -> Dict:
        """Get a summary of all statistics."""
        return {
            "shape_bias": self.compute_shape_bias(),
            "shape_correct": self.shape_correct,
            "texture_correct": self.texture_correct,
            "both_correct": self.both_correct,
            "neither_correct": self.neither_correct,
            "total": self.total,
            "shape_decisions_fraction": self.shape_correct / self.total if self.total > 0 else 0,
            "texture_decisions_fraction": self.texture_correct / self.total if self.total > 0 else 0,
            "correct_fraction": (self.shape_correct + self.texture_correct) / self.total if self.total > 0 else 0,
        }


# =============================================================================
# Cue-Conflict Evaluator (Main Interface)
# =============================================================================
class CueConflictEvaluator:
    """
    Evaluate shape bias on cue-conflict stimuli.
    
    Example usage:
        evaluator = CueConflictEvaluator(
            model, processor, "/path/to/data/cue-conflict",
            device="cuda", batch_size=32
        )
        results = evaluator.evaluate()
        print(f"Shape bias: {results['shape_bias']:.2%}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        processor,
        cue_conflict_path: str,
        device: str = "cuda",
        batch_size: int = 32,
        num_workers: int = 4,
        aggregation: str = "mean",  # Recommended by paper
    ):
        """
        Args:
            model: PyTorch model that outputs ImageNet 1000-class logits
            processor: HuggingFace processor for preprocessing
            cue_conflict_path: Path to cue-conflict dataset directory
            device: Device to run evaluation on
            batch_size: Batch size for evaluation
            num_workers: Number of data loading workers
            aggregation: Method to aggregate ImageNet probs to 16-class 
                        ("mean" recommended by paper authors)
        """
        self.model = model.to(device)
        self.processor = processor
        self.cue_conflict_path = cue_conflict_path
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.aggregation = aggregation
        
        # Build transform
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    
    @torch.no_grad()
    def evaluate(self, verbose: bool = True) -> Dict:
        """
        Run full cue-conflict evaluation.
        
        Args:
            verbose: Whether to print progress
            
        Returns:
            Dictionary with shape bias results
        """
        self.model.eval()
        
        # Create dataset (excluding non-conflict images)
        dataset = CueConflictDataset(
            self.cue_conflict_path,
            transform=self.transform,
            include_non_conflict=False,  # Only conflict images
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        calculator = ShapeBiasCalculator()
        
        # Store detailed results
        all_predictions = []
        
        for batch in tqdm(loader, desc="Evaluating cue-conflict", disable=not verbose):
            pixel_values = batch["pixel_values"].to(self.device)
            shape_classes = batch["shape_class"]
            texture_classes = batch["texture_class"]
            
            # Forward pass
            outputs = self.model(pixel_values)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            elif hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
            # Process each sample
            for i in range(len(probs)):
                prob = probs[i]
                shape_cls = shape_classes[i]
                texture_cls = texture_classes[i]
                
                # Map to 16-class prediction
                predicted_class, class_probs = probabilities_to_16class(
                    prob, aggregation=self.aggregation
                )
                
                # Update calculator
                calculator.update(predicted_class, shape_cls, texture_cls)
                
                all_predictions.append({
                    "shape_class": shape_cls,
                    "texture_class": texture_cls,
                    "predicted_class": predicted_class,
                    "correct_type": "shape" if predicted_class == shape_cls 
                                   else ("texture" if predicted_class == texture_cls else "neither"),
                })
        
        # Compute results
        summary = calculator.get_summary()
        per_class_bias = calculator.compute_per_class_shape_bias()
        
        results = {
            **summary,
            "per_class_shape_bias": per_class_bias,
            "aggregation_method": self.aggregation,
            "num_samples": len(dataset),
        }
        
        if verbose:
            self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict):
        """Print formatted results."""
        print("\n" + "="*60)
        print("CUE-CONFLICT SHAPE BIAS RESULTS")
        print("="*60)
        print(f"Shape Bias: {results['shape_bias']:.2%}")
        print("-"*60)
        print(f"  Shape correct (only):   {results['shape_correct']:>5}")
        print(f"  Texture correct (only): {results['texture_correct']:>5}")
        print(f"  Both correct:           {results['both_correct']:>5}")
        print(f"  Neither correct:        {results['neither_correct']:>5}")
        print(f"  Total:                  {results['total']:>5}")
        print("-"*60)
        print(f"Aggregation method: {results['aggregation_method']}")
        print("="*60)
        
        # Per-class breakdown
        print("\nPer-class shape bias:")
        for cls in SIXTEEN_CLASSES:
            bias = results["per_class_shape_bias"][cls]
            bar = "█" * int(bias * 20) + "░" * (20 - int(bias * 20))
            print(f"  {cls:<12} {bar} {bias:.2%}")
        print()


# =============================================================================
# Convenience function for integration with existing evaluator
# =============================================================================
def evaluate_cue_conflict_shape_bias(
    model: nn.Module,
    processor,
    data_root: str,
    device: str = "cuda",
    batch_size: int = 32,
    num_workers: int = 4,
) -> Dict[str, float]:
    """
    Convenience function to evaluate shape bias on cue-conflict stimuli.
    
    Args:
        model: PyTorch model
        processor: HuggingFace processor
        data_root: Root data directory containing "cue-conflict" subfolder
        device: Device to use
        batch_size: Batch size
        num_workers: Number of workers
        
    Returns:
        Dictionary with shape_bias and other metrics
    """
    cue_conflict_path = os.path.join(data_root, "shape-stimuli", "cue-conflict")
    
    if not os.path.exists(cue_conflict_path):
        raise FileNotFoundError(
            f"Cue-conflict dataset not found at {cue_conflict_path}. "
            "Download from https://github.com/rgeirhos/texture-vs-shape/tree/master/stimuli/style-transfer-preprocessed-512"
        )
    
    evaluator = CueConflictEvaluator(
        model=model,
        processor=processor,
        cue_conflict_path=cue_conflict_path,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    return evaluator.evaluate()


if __name__ == "__main__":
    # Example usage / testing
    print("Cue-Conflict Shape Bias Evaluator")
    print("="*60)
    print("This module provides:")
    print("  - CueConflictDataset: Dataset class for cue-conflict images")
    print("  - CueConflictEvaluator: Full evaluation pipeline")
    print("  - evaluate_cue_conflict_shape_bias: Convenience function")
    print()
    print("16 Entry-Level Categories:")
    for i, cls in enumerate(SIXTEEN_CLASSES):
        print(f"  {i:2d}. {cls}")

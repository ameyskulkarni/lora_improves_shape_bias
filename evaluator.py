"""Evaluation on ImageNet variants and shape bias measurement."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional
import json
import wandb

from datasets import ImageNetDataset, get_transforms


# ImageNet-C corruptions
CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",  # noise
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",  # blur
    "snow", "frost", "fog", "brightness",  # weather
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",  # digital
]


class ShapeBiasEvaluator:
    """Evaluate model on multiple datasets and compute shape bias."""
    
    def __init__(
        self,
        model: nn.Module,
        processor,
        data_root: str,
        device: str = "cuda",
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        self.model = model.to(device)
        self.processor = processor
        self.data_root = data_root
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = get_transforms(processor, is_train=False)
    
    def _get_loader(self, variant: str, **kwargs) -> DataLoader:
        ds = ImageNetDataset(
            self.data_root, variant=variant, split="val",
            transform=self.transform, **kwargs
        )
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )
    
    @torch.no_grad()
    def _evaluate_loader(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        correct, total = 0, 0
        
        for batch in tqdm(loader, leave=False):
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(pixel_values)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)
        
        return {"accuracy": correct / total, "correct": correct, "total": total}
    
    def evaluate_imagenet(self) -> Dict[str, float]:
        """Evaluate on clean ImageNet."""
        loader = self._get_loader("imagenet")
        print(f"Evaluating on test set - IMAGENET")
        return self._evaluate_loader(loader)
    
    def evaluate_stylized(self) -> Dict[str, float]:
        """Evaluate on Stylized ImageNet."""
        loader = self._get_loader("stylized_imagenet")
        print(f"Evaluating on test set - STYLIZED_IMAGENET")
        return self._evaluate_loader(loader)
    
    def evaluate_imagenet_c(self, severities: List[int] = [3]) -> Dict[str, float]:
        """Evaluate on ImageNet-C corruptions."""
        print(f"Evaluating on test set - IMAGENET-C")
        results = {}
        
        for corruption in tqdm(CORRUPTIONS, desc="ImageNet-C"):
            for severity in severities:
                loader = self._get_loader(
                    "imagenet_c", corruption_type=corruption, severity=severity
                )
                metrics = self._evaluate_loader(loader)
                results[f"{corruption}_s{severity}"] = metrics["accuracy"]
        
        results["mean"] = sum(results.values()) / len(results)
        return results
    
    def evaluate_cue_conflict(self) -> Dict[str, float]:
        """Evaluate on cue-conflict stimuli for shape bias."""
        # Cue-conflict has shape_label and texture_label in filename
        # Format: {shape_class}_{texture_class}_{idx}.png
        print(f"Evaluating on test set - IMAGENET-CUE-CONFLICT")
        loader = self._get_loader("cue_conflict")
        
        self.model.eval()
        shape_correct, texture_correct, total = 0, 0, 0
        
        for batch in tqdm(loader, desc="Cue-conflict", leave=False):
            pixel_values = batch["pixel_values"].to(self.device)
            # Labels from ImageFolder are shape labels (folder structure)
            shape_labels = batch["labels"].to(self.device)
            
            outputs = self.model(pixel_values)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            preds = logits.argmax(-1)
            
            shape_correct += (preds == shape_labels).sum().item()
            total += shape_labels.size(0)
        
        shape_bias = shape_correct / total
        return {
            "shape_accuracy": shape_bias,
            "shape_bias": shape_bias,  # Fraction of shape-based decisions
        }

    def evaluate_imagenetv2(self) -> Dict[str, float]:
        """Evaluate on ImageNet-V2."""
        print(f"Evaluating on test set - IMAGENETV2")
        loader = self._get_loader("imagenetv2")
        return self._evaluate_loader(loader)

    def evaluate_imagenet_sketch(self) -> Dict[str, float]:
        """Evaluate on ImageNet-Sketch."""
        print(f"Evaluating on test set - IMAGENET-SKETCH")
        loader = self._get_loader("imagenet_sketch")
        return self._evaluate_loader(loader)

    def full_evaluation(self, log_wandb: bool = True, prefix: str = "eval") -> Dict[str, any]:
        """Run full evaluation suite.

        Args:
            log_wandb: Whether to log to wandb
            prefix: Prefix for wandb logging (e.g., "before", "after")
        """
        # results = {
        #     "imagenet": self.evaluate_imagenet(),
        #     "imagenetv2": self.evaluate_imagenetv2(),
        #     "stylized_imagenet": self.evaluate_stylized(),
        #     "imagenet_sketch": self.evaluate_imagenet_sketch(),
        #     "imagenet_c": self.evaluate_imagenet_c(),
        #     "cue_conflict": self.evaluate_cue_conflict(),
        # }
        results = {
            "imagenet": self.evaluate_imagenet(),
            "imagenetv2": self.evaluate_imagenetv2(),
            "stylized_imagenet": self.evaluate_stylized(),
            "imagenet_sketch": self.evaluate_imagenet_sketch(),
            "imagenet_c": self.evaluate_imagenet_c(),
        }

        # Summary metrics
        # results["summary"] = {
        #     "imagenet_acc": results["imagenet"]["accuracy"],
        #     "imagenetv2_acc": results["imagenetv2"]["accuracy"],
        #     "stylized_acc": results["stylized_imagenet"]["accuracy"],
        #     "imagenet_sketch_acc": results["imagenet_sketch"]["accuracy"],
        #     "imagenet_c_mean": results["imagenet_c"]["mean"],
        #     "shape_bias": results["cue_conflict"]["shape_bias"],
        # }
        results["summary"] = {
            "imagenet_acc": results["imagenet"]["accuracy"],
            "imagenetv2_acc": results["imagenetv2"]["accuracy"],
            "stylized_acc": results["stylized_imagenet"]["accuracy"],
            "imagenet_sketch_acc": results["imagenet_sketch"]["accuracy"],
            "imagenet_c_mean": results["imagenet_c"]["mean"],
        }

        if log_wandb:
            wandb.log({f"{prefix}/{k}": v for k, v in results["summary"].items()})

        return results

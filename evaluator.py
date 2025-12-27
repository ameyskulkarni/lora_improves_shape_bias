"""Evaluation on ImageNet variants and shape bias measurement.

This module provides comprehensive evaluation including:
- Standard ImageNet accuracy
- ImageNet-V2 (robustness to distribution shift)
- ImageNet-C (corruption robustness)
- ImageNet-Sketch (sketch recognition)
- Stylized-ImageNet (texture-removed)
- Cue-Conflict Shape Bias (from Geirhos et al., ICLR 2019)
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional
import wandb
import numpy as np
from torchvision import transforms

from datasets import ImageNetDataset, get_transforms, ShapeStimuliDataset

from cue_conflict_evaluator import (
    CueConflictEvaluator,
    IMAGENET_TO_16CLASS_MAPPING,
    SIXTEEN_CLASSES,
    CLASS_TO_IDX,
)


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

    @torch.no_grad()
    def _evaluate_loader_16class(self, loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate using 16-class aggregation (for edge, silhouette).

        Maps ImageNet-1000 probabilities to 16 categories using averaging,
        following the model-vs-human methodology:
        "We here used the average: ImageNet class probabilities were mapped to the
        corresponding 16-class-ImageNet category using the average of all
        corresponding fine-grained category probabilities."
        """
        self.model.eval()
        correct, total = 0, 0

        for batch in tqdm(loader, leave=False):
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)  # These are 0-15 (16 categories)

            # Get 1000-class logits
            outputs = self.model(pixel_values)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits

            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # For each sample, aggregate to 16 classes
            batch_size = probs.shape[0]
            predictions_16class = np.zeros(batch_size, dtype=np.int64)

            for i in range(batch_size):
                # Aggregate ImageNet-1000 probs to 16-class probs using mean
                class_probs_16 = np.zeros(16)

                for class_idx, class_name in enumerate(SIXTEEN_CLASSES):
                    imagenet_indices = IMAGENET_TO_16CLASS_MAPPING[class_name]
                    # Average probabilities of all ImageNet classes in this category
                    class_probs_16[class_idx] = np.mean(probs[i, imagenet_indices])

                # Get predicted category (0-15)
                predictions_16class[i] = np.argmax(class_probs_16)

            # Compare predictions with ground truth
            predictions_16class = torch.from_numpy(predictions_16class).to(self.device)
            correct += (predictions_16class == labels).sum().item()
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
        """
        Evaluate on cue-conflict stimuli for shape bias.

        This implements the methodology from:
        "ImageNet-trained CNNs are biased towards texture" (Geirhos et al., ICLR 2019)

        The shape bias is computed as:
            shape_bias = shape_correct / (shape_correct + texture_correct)

        Where:
        - shape_correct: model predicted the shape category
        - texture_correct: model predicted the texture category
        - Only "correctly" classified images are counted (predicted shape OR texture)
        - Non-conflict images (shape == texture) are excluded

        Returns:
            Dictionary with shape_bias and detailed statistics
        """
        print(f"Evaluating on test set - CUE-CONFLICT SHAPE BIAS")

        cue_conflict_path = os.path.join(self.data_root, "shape-stimuli", "cue-conflict")

        if not os.path.exists(cue_conflict_path):
            print(f"  Warning: Cue-conflict dataset not found at {cue_conflict_path}")
            print("  Download from: https://github.com/rgeirhos/texture-vs-shape/tree/master/stimuli/style-transfer-preprocessed-512")
            return {"shape_bias": None}

        evaluator = CueConflictEvaluator(
            model=self.model,
            processor=self.processor,
            cue_conflict_path=cue_conflict_path,
            device=self.device,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            aggregation="mean",  # Recommended by paper authors
        )

        results = evaluator.evaluate(verbose=False)

        # Print summary
        print(f"  Shape Bias: {results['shape_bias']:.2%}")
        print(f"  Shape correct: {results['shape_correct']}, "
              f"Texture correct: {results['texture_correct']}")

        return results

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

    def evaluate_shape_stimuli_dataset(self, dataset_name: str) -> Dict[str, float]:
        """
        Evaluate on a single shape-stimuli dataset.

        For edge and silhouette: Uses 16-class aggregation (averaging method)
        For other datasets: Uses standard ImageNet-1000 evaluation

        Args:
            dataset_name: One of 'edge', 'silhouette', 'color', 'contrast',
                         'high-pass', 'low-pass', 'phase-scrambling',
                         'power-equalization', 'false-color', 'rotation',
                         'eidolonI', 'eidolonII', 'eidolonIII', 'uniform-noise', 'sketch'

        Returns:
            Dictionary with accuracy metrics
        """
        shape_stimuli_path = os.path.join(self.data_root, "shape-stimuli")

        if not os.path.exists(shape_stimuli_path):
            print(f"  Warning: shape-stimuli folder not found at {shape_stimuli_path}")
            return {"accuracy": None}

        # Datasets that need 16-class aggregation
        USE_16_CLASS_AGGREGATION = {'edge', 'silhouette'}

        try:
            # Use ImageNet normalization
            IMAGENET_MEAN = [0.485, 0.456, 0.406]
            IMAGENET_STD = [0.229, 0.224, 0.225]
            shape_stimuli_transform = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                                    ])
            dataset = ShapeStimuliDataset(
                root=shape_stimuli_path,
                dataset_name=dataset_name,
                transform=shape_stimuli_transform,
            )

            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            print(f"Evaluating on test set - SHAPE-STIMULI/{dataset_name.upper()}")
            return self._evaluate_loader_16class(loader)

            # Use 16-class aggregation for edge and silhouette
            # if dataset_name in USE_16_CLASS_AGGREGATION:
            #     print(f"  Using 16-class aggregation (averaging)")
            #     return self._evaluate_loader_16class(loader)
            # else:
            #     return self._evaluate_loader(loader)

        except Exception as e:
            print(f"  Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return {"accuracy": None}

    def evaluate_all_shape_stimuli(self) -> Dict[str, float]:
        """
        Evaluate on all 15 shape-stimuli datasets.

        Returns:
            Dictionary with accuracy for each dataset
        """
        datasets = [
            'edge', 'silhouette', 'color', 'contrast',
            'high-pass', 'low-pass', 'phase-scrambling', 'power-equalization',
            'false-color', 'rotation', 'eidolonI', 'eidolonII', 'eidolonIII',
            'uniform-noise', 'sketch', 'stylized'
        ]

        results = {}
        for dataset_name in datasets:
            result = self.evaluate_shape_stimuli_dataset(dataset_name)
            if result['accuracy'] is not None:
                results[f"{dataset_name}_acc"] = result['accuracy']

        # Compute mean across available datasets
        if results:
            results['mean_acc'] = sum(results.values()) / len(results)

        return results

    def full_evaluation(
        self,
        log_wandb: bool = True,
        prefix: str = "eval",
        include_shape_bias: bool = True,
        include_shape_stimuli: bool = True,
    ) -> Dict[str, any]:
        """Run full evaluation suite.

        Args:
            log_wandb: Whether to log to wandb
            prefix: Prefix for wandb logging (e.g., "before", "after")
            include_shape_bias: Whether to include cue-conflict shape bias evaluation
            include_shape_stimuli: Whether to include all 15 shape-stimuli datasets
        """
        results = {}
        summary = {}
        # results = {
        #     "imagenet": self.evaluate_imagenet(),
        #     "imagenetv2": self.evaluate_imagenetv2(),
        #     "stylized_imagenet": self.evaluate_stylized(),
        #     "imagenet_sketch": self.evaluate_imagenet_sketch(),
        #     "imagenet_c": self.evaluate_imagenet_c(),
        # }

        # Add cue-conflict shape bias if requested
        if include_shape_bias:
            cue_conflict_results = self.evaluate_cue_conflict()
            results["cue_conflict"] = cue_conflict_results
            if cue_conflict_results.get("shape_bias") is not None:
                summary["shape_bias"] = cue_conflict_results["shape_bias"]

        # Add shape-stimuli datasets if requested
        if include_shape_stimuli:
            shape_stimuli_results = self.evaluate_all_shape_stimuli()
            results["shape_stimuli"] = shape_stimuli_results
            # Add each dataset accuracy to summary
            for key, value in shape_stimuli_results.items():
                if value is not None:
                    summary[f"shape_stimuli/{key}"] = value

        # Summary metrics
        # summary = {
        #     "imagenet_acc": results["imagenet"]["accuracy"],
        #     "imagenetv2_acc": results["imagenetv2"]["accuracy"],
        #     "stylized_acc": results["stylized_imagenet"]["accuracy"],
        #     "imagenet_sketch_acc": results["imagenet_sketch"]["accuracy"],
        #     "imagenet_c_mean": results["imagenet_c"]["mean"],
        # }

        results["summary"] = summary

        if log_wandb:
            wandb.log({f"{prefix}/{k}": v for k, v in results["summary"].items()})

        return results

    # def evaluate_shape_bias_only(self, log_wandb: bool = True, prefix: str = "eval") -> Dict:
    #     """
    #     Run only the cue-conflict shape bias evaluation.
    #
    #     Useful for quick shape bias checks without full evaluation.
    #     """
    #     results = self.evaluate_cue_conflict()
    #
    #     if log_wandb and results.get("shape_bias") is not None:
    #         wandb.log({f"{prefix}/shape_bias": results["shape_bias"]})
    #
    #     return results


# # =============================================================================
# # Standalone evaluation function
# # =============================================================================
# def evaluate_model_shape_bias(
#     model: nn.Module,
#     processor,
#     data_root: str,
#     device: str = "cuda",
#     batch_size: int = 64,
#     num_workers: int = 4,
#     verbose: bool = True,
# ) -> Dict[str, float]:
#     """
#     Standalone function to evaluate shape bias of any model.
#
#     Args:
#         model: PyTorch model that outputs ImageNet 1000-class logits
#         processor: HuggingFace processor for preprocessing
#         data_root: Root data directory (should contain "cue-conflict" subfolder)
#         device: Device to use
#         batch_size: Batch size for evaluation
#         num_workers: Number of data loading workers
#         verbose: Whether to print results
#
#     Returns:
#         Dictionary with shape_bias and related metrics
#     """
#     cue_conflict_path = os.path.join(data_root, 'shape-stimuli', "cue-conflict")
#
#     if not os.path.exists(cue_conflict_path):
#         raise FileNotFoundError(
#             f"Cue-conflict dataset not found at {cue_conflict_path}.\n"
#             "Download from: https://github.com/rgeirhos/texture-vs-shape/tree/master/stimuli/style-transfer-preprocessed-512"
#         )
#
#     evaluator = CueConflictEvaluator(
#         model=model,
#         processor=processor,
#         cue_conflict_path=cue_conflict_path,
#         device=device,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         aggregation="mean",
#     )
#
#     return evaluator.evaluate(verbose=verbose)
"""ViT with LoRA adapters for shape bias training."""
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import LoraConfig, get_peft_model


def get_lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.1) -> LoraConfig:
    """LoRA config targeting all attention + MLP layers."""
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=[
            "query", "key", "value",
            "dense", "intermediate.dense", "output.dense",
        ],
    )


def create_vit_lora(
    model_name: str = "facebook/deit-tiny-patch16-224",
    num_classes: int = 1000,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    freeze_base: bool = True,
) -> tuple[nn.Module, ViTImageProcessor]:
    """Create ViT-Tiny with LoRA adapters."""
    model = ViTForImageClassification.from_pretrained(
        model_name, num_labels=num_classes, ignore_mismatched_sizes=True,
    )
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
    
    lora_config = get_lora_config(r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, processor


class ShapeBiasViT(nn.Module):
    """Wrapper for future multi-task extension (Phase 2)."""
    
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        self.aux_heads = nn.ModuleDict()
    
    def forward(self, pixel_values: torch.Tensor, return_features: bool = False):
        outputs = self.base(pixel_values, output_hidden_states=return_features)
        result = {"logits": outputs.logits}
        if return_features:
            result["features"] = outputs.hidden_states[-1][:, 0]
        return result
    
    def add_aux_head(self, name: str, head: nn.Module):
        self.aux_heads[name] = head

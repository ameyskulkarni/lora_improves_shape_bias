"""Training loop for LoRA fine-tuning."""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Optional, Dict, Any
import wandb
from peft import PeftModel


class LoRATrainer:
    """Trainer for LoRA fine-tuning with shape bias objective."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: Dict[str, Any],
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 0.05),
        )

        # Warmup + Cosine Annealing scheduler
        total_steps = len(train_loader) * config.get("epochs", 10)
        warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))

        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps
        )
        self.scheduler = SequentialLR(
            self.optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        )

        self.scaler = GradScaler() if config.get("fp16", True) else None
        self.global_step = 0
        self.best_acc = 0.0

        # Log all config to wandb in one place
        self._log_config()

    def _log_config(self):
        """Log all training configuration to wandb."""
        wandb.config.update({
            "model": self.config.get("model_name", "facebook/deit-tiny-patch16-224"),
            "lora_r": self.config.get("lora_r", 16),
            "lora_alpha": self.config.get("lora_alpha", 32),
            "lr": self.config.get("lr", 1e-4),
            "weight_decay": self.config.get("weight_decay", 0.05),
            "epochs": self.config.get("epochs", 10),
            "batch_size": self.config.get("batch_size", 64),
            "warmup_ratio": self.config.get("warmup_ratio", 0.1),
            "fp16": self.config.get("fp16", True),
            "train_variant": self.config.get("train_variant", "stylized_imagenet"),
            "val_variant": self.config.get("val_variant", "imagenet"),
            "save_every": self.config.get("save_every", None),
        })

    def train_epoch(self, epoch: int, save_dir: Optional[str] = None) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        save_every = self.config.get("save_every")

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            if self.scaler:
                with autocast():
                    outputs = self.model(pixel_values)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(pixel_values)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()  # Step per iteration for warmup

            total_loss += loss.item()
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)
            self.global_step += 1

            pbar.set_postfix(loss=loss.item(), acc=correct/total)

            if self.global_step % self.config.get("log_every", 100) == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/acc": correct / total,
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/epoch": epoch,
                }, step=self.global_step)

            # Interval checkpointing
            if save_every and save_dir and self.global_step % save_every == 0:
                self.save_checkpoint(save_dir, tag=f"step_{self.global_step}")

        return {"loss": total_loss / len(self.train_loader), "acc": correct / total}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(self.val_loader, desc="Validation"):
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(pixel_values)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)

        metrics = {"loss": total_loss / len(self.val_loader), "acc": correct / total}
        wandb.log({"val/loss": metrics["loss"], "val/acc": metrics["acc"]}, step=self.global_step)
        return metrics

    def train(self, epochs: int, save_dir: Optional[str] = None):
        _ = self.validate()
        for epoch in range(epochs):
            train_metrics = self.train_epoch(epoch, save_dir=save_dir)
            val_metrics = self.validate()

            print(f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                  f"Val Acc={val_metrics['acc']:.4f}")

            # Save best checkpoint
            if val_metrics["acc"] > self.best_acc and save_dir:
                self.best_acc = val_metrics["acc"]
                self.save_checkpoint(save_dir, tag="best_checkpoint")
                wandb.run.summary["best_val_acc"] = self.best_acc

            # Save epoch checkpoint
            if save_dir:
                self.save_checkpoint(save_dir, tag=f"epoch_{epoch}")

        return {"best_val_acc": self.best_acc}

    def save_checkpoint(self, save_dir: str, tag: str = "latest"):
        """Save LoRA weights with a tag."""
        path = os.path.join(save_dir, tag)
        os.makedirs(path, exist_ok=True)
        self.model.base.save_pretrained(path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load LoRA weights from checkpoint."""
        base = self.model.base.get_base_model()
        self.model.base = PeftModel.from_pretrained(base, path)
        self.model.to(self.device)
        print(f"Loaded checkpoint from {path}")
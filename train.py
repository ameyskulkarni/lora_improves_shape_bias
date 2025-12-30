"""Train LoRA for shape bias on Stylized ImageNet."""
import argparse
import os
import torch

import torch
import wandb
from peft import PeftModel

from vit_lora import create_vit_lora, ShapeBiasViT
from datasets import create_dataloaders_mixed
from lora_trainer import LoRATrainer
from evaluator import ShapeBiasEvaluator
from seed_setting import set_seed, get_generator


def parse_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--train-variant", type=str, default="imagenet", help="Can be specific datasets or 'mixed_stylized', 'mixed_depth', 'mixed_all', 'depth'. mixed_stylized combines imagenet+stylized_imagenet")
    parser.add_argument("--val-variant", type=str, default="imagenet", help="Can be specific datasets or 'mixed_stylized', 'mixed_depth', 'mixed_all', 'depth'. mixed_stylized combines imagenet+stylized_imagenet")
    # Model
    parser.add_argument("--model-name", type=str, default="facebook/deit-tiny-patch16-224")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    # Training
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    # Logging & Checkpointing
    parser.add_argument("--wandb-project", type=str, default="lora-shape-bias")
    parser.add_argument("--run-name", type=str, default="deit-tiny-stylized")
    parser.add_argument("--tags", type=str, default="",
                        help="Comma-separated tags (e.g., baseline,experiment1)")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=None, help="Save checkpoint every N steps")
    # Eval
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--skip-before-eval", action="store_true", help="Skip evaluation before training")
    return parser.parse_args()

def print_comparison(before: dict, after: dict):
    """Print before/after comparison table."""
    print("\n" + "="*60)
    print(f"{'Metric':<25} {'Before':>12} {'After':>12} {'Δ':>10}")
    print("="*60)
    for key in before["summary"]:
        b = before["summary"][key]
        a = after["summary"][key]
        delta = a - b
        sign = "+" if delta > 0 else ""
        print(f"{key:<25} {b:>12.4f} {a:>12.4f} {sign}{delta:>9.4f}")
    print("="*60 + "\n")

def main():
    args = parse_args()

    # SET SEED FIRST (before any random operations)
    seed_worker = set_seed(args.seed, deterministic=True)
    g = get_generator(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create save directory: checkpoints/{run_name}/
    save_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(save_dir, exist_ok=True)


    # Init wandb
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        tags=[tag.strip() for tag in args.tags.split(",") if tag.strip()],
        config={
            **vars(args),  # Includes seed
            "seed": args.seed,
        },
    )
    
    # Create model
    print("Creating model...")
    base_model, processor = create_vit_lora(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = ShapeBiasViT(base_model)
    
    # Load checkpoint if provided
    if args.checkpoint:
        base = model.base.get_base_model()
        model.base = PeftModel.from_pretrained(base, args.checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    # Create dataloaders
    print(f"Loading data: train={args.train_variant}, val={args.val_variant}")
    train_loader, val_loader = create_dataloaders_mixed(
        args.data_root, processor,
        train_variant=args.train_variant,  # stylized
        val_variant=args.val_variant,  # normal ImageNet for validation
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        generator=g,
        worker_init_fn=seed_worker,
    )
    
    if args.eval_only:
        evaluator = ShapeBiasEvaluator(
            model, processor, args.data_root,
            device=device, batch_size=args.batch_size,
        )
        results = evaluator.full_evaluation()
        print("\n=== Evaluation Results ===")
        for k, v in results["summary"].items():
            print(f"{k}: {v:.4f}")
        wandb.finish()
        return

    # Run BEFORE evaluation (baseline)
    before_results = None
    if not args.skip_before_eval:
        evaluator = ShapeBiasEvaluator(
            model, processor, args.data_root,
            device=device, batch_size=args.batch_size,
        )
        print("\n=== Evaluating BEFORE training (baseline) ===")
        before_results = evaluator.full_evaluation(prefix="before")
        print("Before training metrics:")
        for k, v in before_results["summary"].items():
            print(f"  {k}: {v:.4f}")

    # Train
    config = {
        "model_name": args.model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "fp16": args.fp16,
        "log_every": args.log_every,
        "save_every": args.save_every,
        "train_variant": args.train_variant,
        "val_variant": args.val_variant,
    }
    
    trainer = LoRATrainer(model, train_loader, val_loader, config, device, evaluator_args=(processor, args.data_root, args.batch_size))
    
    print("Starting training...")
    trainer.train(args.epochs, save_dir=save_dir)

    # Load best checkpoint for final evaluation
    best_ckpt = os.path.join(save_dir, "best_checkpoint")
    if os.path.exists(best_ckpt):
        print(f"\nLoading best checkpoint from {best_ckpt}")
        trainer.load_checkpoint(best_ckpt)

    # Final evaluation
    print("\nRunning final evaluation...")
    evaluator = ShapeBiasEvaluator(
        trainer.model, processor, args.data_root,
        device=device, batch_size=args.batch_size,
    )
    after_results = evaluator.full_evaluation(prefix="after")

    # Print comparison
    if before_results:
        print_comparison(before_results, after_results)

        # Log deltas to wandb
        for key in before_results["summary"]:
            delta = after_results["summary"][key] - before_results["summary"][key]
            wandb.run.summary[f"delta/{key}"] = delta
    else:
        print("\n=== Final Results (After Training) ===")
        for k, v in after_results["summary"].items():
            print(f"{k}: {v:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()

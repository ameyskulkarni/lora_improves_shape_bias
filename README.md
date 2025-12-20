# Shape Bias LoRA: Increasing Shape Bias in Vision Transformers

Fine-tune Vision Transformers using LoRA to increase shape bias over texture bias, improving robustness and alignment with human visual perception.

## Motivation

Neural networks trained on ImageNet exhibit a strong **texture bias**—they rely heavily on local texture patterns rather than global shape information. This differs from human vision, which is predominantly shape-based. Models with higher shape bias show:

- Better robustness to distribution shifts
- Improved performance on corrupted images
- More human-aligned error patterns

This project uses **Stylized ImageNet** (texture-randomized images) to train LoRA adapters that shift the model toward shape-based representations.

---

## Architecture

### Base Model: ViT-Tiny

```
Input Image (224×224×3)
        │
        ▼
┌─────────────────┐
│  Patch Embed    │  Split into 14×14 = 196 patches (16×16 each)
│  (16×16 stride) │  Project to 192-dim embeddings
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  [CLS] + Patch  │  Prepend learnable [CLS] token
│    Tokens       │  Add position embeddings
│   (197 × 192)   │  
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│         Transformer Encoder             │
│            (12 Blocks)                  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  Multi-Head Self-Attention (3 heads) │
│  │  ┌─────┐ ┌─────┐ ┌─────┐         │  │
│  │  │  Q  │ │  K  │ │  V  │ ◄─ LoRA │  │
│  │  └──┬──┘ └──┬──┘ └──┬──┘         │  │
│  │     └───────┼───────┘            │  │
│  │             ▼                    │  │
│  │     ┌─────────────┐              │  │
│  │     │   Attention │              │  │
│  │     │   Output    │ ◄─────LoRA   │  │
│  │     └─────────────┘              │  │
│  └───────────────────────────────────┘  │
│                  │                      │
│                  ▼                      │
│  ┌───────────────────────────────────┐  │
│  │           MLP Block               │  │
│  │  ┌─────────────┐                  │  │
│  │  │ Intermediate│ 192→768 ◄─ LoRA  │  │
│  │  │   (up-proj) │                  │  │
│  │  └──────┬──────┘                  │  │
│  │         ▼                         │  │
│  │      [GELU]                       │  │
│  │         ▼                         │  │
│  │  ┌─────────────┐                  │  │
│  │  │   Output    │ 768→192 ◄─ LoRA  │  │
│  │  │ (down-proj) │                  │  │
│  │  └─────────────┘                  │  │
│  └───────────────────────────────────┘  │
│                                         │
│              × 12 blocks                │
└────────────────────┬────────────────────┘
                     │
                     ▼
              ┌─────────────┐
              │ [CLS] Token │  Extract final [CLS] representation
              │   (192-d)   │
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │ Classifier  │  Linear: 192 → 1000 classes
              │    Head     │
              └─────────────┘
```

### LoRA Injection Points

LoRA (Low-Rank Adaptation) adds trainable low-rank matrices while keeping the base model frozen:

```
Original:       h = W₀x
With LoRA:      h = W₀x + (α/r) · BAx

Where:
  W₀ ∈ ℝ^(d×k)  - Frozen pretrained weights  
  A ∈ ℝ^(r×k)   - Trainable down-projection (init: Gaussian)
  B ∈ ℝ^(d×r)   - Trainable up-projection (init: zeros)
  r = 16        - LoRA rank (compression factor)
  α = 32        - LoRA scaling factor
```

**Target Modules** (per transformer block):
| Module | Dimensions | LoRA Params |
|--------|-----------|-------------|
| `query` | 192 → 192 | 2 × 192 × 16 = 6,144 |
| `key` | 192 → 192 | 6,144 |
| `value` | 192 → 192 | 6,144 |
| `attention.output.dense` | 192 → 192 | 6,144 |
| `intermediate.dense` | 192 → 768 | (192+768) × 16 = 15,360 |
| `output.dense` | 768 → 192 | 15,360 |

**Total LoRA Parameters:** ~660K (vs 5.7M base) = **11.5% of original**

---

## Installation

```bash
git clone <repo>
cd lora_improves_shape_bias
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py \
    --data_root /path/to/data \
    --variant stylized \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --wandb_project shape-bias-lora
```

### Evaluation Only

```bash
python scripts/train.py \
    --eval_only \
    --checkpoint checkpoints/lora_stylized \
    --data_root /path/to/data
```

---

## Evaluation Metrics

| Metric | Dataset | Measures |
|--------|---------|----------|
| **ImageNet Acc** | ImageNet-Val | Standard accuracy (texture+shape) |
| **Stylized Acc** | Stylized-ImageNet | Shape-only accuracy |
| **ImageNet-C mCE** | ImageNet-C | Corruption robustness |
| **ImageNet-V2 Acc** | ImageNet-V2 | Distribution shift robustness |
| **Shape Bias** | Cue-Conflict | % of shape-based decisions |

**Shape Bias Interpretation:**
- Baseline CNNs: ~20-30% (texture-biased)
- Humans: ~95% (shape-biased)  
- Goal: Push ViT toward higher shape bias (>50%)

---

## Phase 2 Roadmap

The codebase is designed for extension with auxiliary tasks:

```python
# Add depth prediction head
from models.vit_lora import ShapeBiasViT

model = ShapeBiasViT(base_model)
model.add_aux_head("depth", DepthHead(192, output_size=224))

# Forward with features
outputs = model(images, return_features=True)
cls_features = outputs["features"]  # For auxiliary tasks
```

---

## References

- [Geirhos et al. (2019) - ImageNet-trained CNNs are biased towards texture](https://arxiv.org/abs/1811.12231)
- [Hu et al. (2021) - LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hendrycks & Dietterich (2019) - Benchmarking Neural Network Robustness to Common Corruptions](https://arxiv.org/abs/1903.12261)

## License

MIT

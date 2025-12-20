# Depth Map Generation for ImageNet

These scripts generate and visualize depth maps for ImageNet using Depth Anything V2.

## Installation

```bash
pip install torch torchvision transformers pillow matplotlib numpy tqdm
```

## Usage

### 1. Generate Depth Maps

Generate depth maps for validation set:
```bash
python generate_depth.py --data-root /path/to/data --split val
```

Generate depth maps for training set:
```bash
python generate_depth.py --data-root /path/to/data --split train
```

**Options:**
- `--data-root`: Path to your data directory (should contain `imagenet/train` or `imagenet/val`)
- `--split`: Choose `train` or `val` (default: `val`)
- `--model-name`: Choose model size (default: `depth-anything/Depth-Anything-V2-Small-hf`)
  - Options: `Depth-Anything-V2-Small-hf`, `Depth-Anything-V2-Base-hf`, `Depth-Anything-V2-Large-hf`
- `--save-as-numpy`: Save as `.npy` files instead of PNG (preserves float precision)
- `--device`: Use `cuda` or `cpu`

**Output Structure:**
```
data_root/
├── imagenet/
│   ├── train/
│   └── val/
└── imagenet_depth/  # Generated depth maps
    ├── train/
    │   ├── n01440764/
    │   │   ├── image1.png
    │   │   └── image2.png
    │   └── ...
    └── val/
        └── ...
```

### 2. Visualize RGB-Depth Pairs

**Side-by-side view (5 samples):**
```bash
python visualize_depth.py --data-root /path/to/data --num-samples 5
```

**Grid view (16 samples):**
```bash
python visualize_depth.py --data-root /path/to/data --num-samples 16 --grid
```

**Save to file:**
```bash
python visualize_depth.py \
    --data-root /path/to/data \
    --num-samples 10 \
    --save-path depth_visualization.png
```

**Visualize specific class:**
```bash
python visualize_depth.py \
    --data-root /path/to/data \
    --class-name n01440764 \
    --num-samples 5
```

**Options:**
- `--data-root`: Path to your data directory
- `--split`: Choose `train` or `val` (default: `val`)
- `--num-samples`: Number of samples to show (default: 5)
- `--random-seed`: Random seed for sampling (default: 42)
- `--save-path`: Save visualization to file instead of displaying
- `--class-name`: Visualize specific ImageNet class (e.g., `n01440764`)
- `--grid`: Use compact grid layout

## Example Workflow

```bash
# 1. Generate depth maps for validation set
python generate_depth.py \
    --data-root /path/to/data \
    --split val \
    --model-name depth-anything/Depth-Anything-V2-Small-hf

# 2. Visualize some examples
python visualize_depth.py \
    --data-root /path/to/data \
    --num-samples 10 \
    --grid \
    --save-path depth_examples.png

# 3. Generate depth for training set (this will take longer)
python generate_depth.py \
    --data-root /path/to/data \
    --split train
```

## Notes

- The script automatically skips already-processed images, so you can safely resume interrupted runs
- For the full ImageNet training set (~1.28M images), this will take several hours on GPU
- PNG format is recommended for most use cases (saves space)
- Use `--save-as-numpy` if you need exact depth values for training

## Model Sizes

| Model | Parameters | Speed | Quality |
|-------|-----------|-------|---------|
| Small | 24.8M | Fast | Good |
| Base | 97.5M | Medium | Better |
| Large | 335.3M | Slow | Best |

For most purposes, the Small model is sufficient and much faster.

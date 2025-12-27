# Cue-Conflict Shape Bias Evaluation

This implementation provides the **exact methodology** for computing shape bias as described in:

> **"ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness"**  
> Geirhos et al., ICLR 2019  
> https://openreview.net/forum?id=Bygh9j09KX

## 📁 Files Created

| File | Description |
|------|-------------|
| `cue_conflict_evaluator.py` | Core implementation of shape bias calculation |
| `evaluator.py` | Updated evaluator with integrated shape bias |

## 🔧 Integration Instructions

### Step 1: Add the new file to your project
Copy `cue_conflict_evaluator.py` to your project root (same directory as `evaluator.py`).

### Step 2: Update your evaluator.py
Replace your existing `evaluator.py` with the new version, OR add the import and method:

```python
# At the top of evaluator.py
from cue_conflict_evaluator import CueConflictEvaluator

# In the ShapeBiasEvaluator class, replace/update evaluate_cue_conflict():
def evaluate_cue_conflict(self) -> Dict[str, float]:
    """Evaluate on cue-conflict stimuli for shape bias."""
    print(f"Evaluating on test set - CUE-CONFLICT SHAPE BIAS")
    
    cue_conflict_path = os.path.join(self.data_root, "cue-conflict")
    
    evaluator = CueConflictEvaluator(
        model=self.model,
        processor=self.processor,
        cue_conflict_path=cue_conflict_path,
        device=self.device,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        aggregation="mean",  # Recommended by paper authors
    )
    
    return evaluator.evaluate(verbose=False)
```

### Step 3: Ensure your data structure
Your data directory should contain:
```
data_root/
├── cue-conflict/          # <-- The cue-conflict dataset
│   ├── airplane/
│   │   ├── airplane1-bear2.png
│   │   ├── airplane1-bicycle3.png
│   │   └── ...
│   ├── bear/
│   ├── bicycle/
│   └── ... (16 folders total)
├── imagenet/
├── stylized_imagenet/
└── ...
```

**Download the cue-conflict dataset from:**
https://github.com/rgeirhos/texture-vs-shape/tree/master/stimuli/style-transfer-preprocessed-512

## 📊 Shape Bias Calculation Methodology

### Official 5-Step Process (from the paper's FAQ)

1. **Evaluate on all 1,280 cue-conflict images**
   - Images have format: `{shape}{num}-{texture}{num}.png`
   - Folder name = shape category
   - Filename second part = texture category

2. **Map ImageNet 1000-class predictions → 16 entry-level categories**
   - Uses **average aggregation** (recommended by authors):
     ```python
     # For each 16-class category, average the probabilities of all
     # corresponding ImageNet classes
     prob_16[cat] = mean(prob_1000[imagenet_indices_for_cat])
     ```

3. **Exclude non-conflict images**
   - Skip images where `shape == texture` (e.g., `cat1-cat2.png`)
   - These don't create a shape-texture conflict

4. **Keep only "correct" predictions**
   - Model must predict EITHER the shape OR texture category
   - Predictions of neither category are excluded from bias calculation

5. **Compute Shape Bias**
   ```
   shape_bias = shape_correct / (shape_correct + texture_correct)
   ```
   - **0.0** = Pure texture bias (always predicts texture)
   - **0.5** = No bias (equal shape/texture predictions)
   - **1.0** = Pure shape bias (always predicts shape)

### The 16 Entry-Level Categories

| Category | ImageNet Class Examples |
|----------|------------------------|
| airplane | airliner |
| bear | brown bear, black bear, polar bear, sloth bear |
| bicycle | bicycle-built-for-two, mountain bike |
| bird | cock, hen, ostrich, crane, flamingo, albatross, ... (many) |
| boat | canoe, fireboat, lifeboat, speedboat, catamaran |
| bottle | beer bottle, pill bottle, pop bottle, water bottle, wine bottle |
| car | beach wagon, cab, convertible, jeep, limousine, sports car |
| cat | tabby, tiger cat, Persian, Siamese, Egyptian, cougar, ... |
| chair | barber chair, folding chair, rocking chair, throne |
| clock | analog clock, digital clock, wall clock |
| dog | all 120 dog breeds in ImageNet |
| elephant | Indian elephant, African elephant |
| keyboard | computer keyboard, typewriter keyboard |
| knife | cleaver |
| oven | Dutch oven, microwave, rotisserie, toaster |
| truck | fire engine, garbage truck, moving van, pickup, ... |

## 📈 Expected Results

Based on the paper and GitHub issues, typical shape bias values:

| Model | Shape Bias | Note |
|-------|-----------|------|
| VGG-16 (ImageNet) | ~9-17% | Strong texture bias |
| AlexNet (ImageNet) | ~25-43% | Moderate texture bias |
| ResNet-50 (ImageNet) | ~21-22% | Moderate texture bias |
| Humans | ~95% | Strong shape bias |
| ResNet-50 (Stylized-IN) | ~81% | Shape-biased after SIN training |

## 🧪 Usage Example

```python
from cue_conflict_evaluator import CueConflictEvaluator

# Create evaluator
evaluator = CueConflictEvaluator(
    model=your_model,
    processor=your_processor,
    cue_conflict_path="/path/to/cue-conflict",
    device="cuda",
    batch_size=32,
)

# Run evaluation
results = evaluator.evaluate()

print(f"Shape Bias: {results['shape_bias']:.2%}")
print(f"Shape correct: {results['shape_correct']}")
print(f"Texture correct: {results['texture_correct']}")
```

## ⚠️ Important Notes for Your Paper

1. **Aggregation Method**: We use **mean aggregation** as recommended by the authors:
   > "We here used the average: ImageNet class probabilities were mapped to the corresponding 16-class-ImageNet category using the average of all corresponding fine-grained category probabilities. We recommend using this approach."

2. **Preprocessing**: Uses standard ImageNet preprocessing:
   - Resize to 256 (if using crop_pct=0.875)
   - Center crop to 224
   - Normalize with ImageNet mean/std

3. **Consistency with Official Repo**: The calculation follows the exact methodology from:
   - https://github.com/rgeirhos/texture-vs-shape
   - https://github.com/bethgelab/model-vs-human

4. **Reproducibility**: 
   - Results may vary slightly from caffe models (original paper used caffe for some models)
   - PyTorch torchvision models may show slightly different bias values
   - This is documented in GitHub issues: https://github.com/rgeirhos/texture-vs-shape/issues/26

## 📚 References

- Paper: https://openreview.net/forum?id=Bygh9j09KX
- Code: https://github.com/rgeirhos/texture-vs-shape
- Benchmark: https://github.com/bethgelab/model-vs-human
- Mapping: https://github.com/rgeirhos/generalisation-humans-DNNs

## 📝 Citation

If using this implementation, please cite the original paper:

```bibtex
@inproceedings{geirhos2019imagenet,
  title={ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness},
  author={Geirhos, Robert and Rubisch, Patricia and Michaelis, Claudio and Bethge, Matthias and Wichmann, Felix A and Brendel, Wieland},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

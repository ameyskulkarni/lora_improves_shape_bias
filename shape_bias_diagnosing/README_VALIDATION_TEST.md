# Shape Bias Validation Test Suite

This test suite validates your cue-conflict shape bias calculations against known baseline models from the Geirhos et al. (ICLR 2019) paper.

## Problem Summary

You're experiencing discrepancies between:
1. **Your codebase**: Shape bias calculated using your `cue_conflict_evaluator.py`
2. **Original repositories**: Results from the official texture-vs-shape and model-vs-human repos

This test suite addresses two potential causes:

### Issue 1: Calculation Methodology
**Solution**: Test against ResNet-50 models with known shape bias values to validate your implementation.

### Issue 2: DeiT Model Source & Preprocessing
**Solution**: Compare DeiT models loaded from torch.hub vs HuggingFace/timm with different normalization.

---

## Quick Start

### Step 1: Download ResNet Checkpoints (Optional but Recommended)

```bash
# Download all official ResNet SIN checkpoints
python download_resnet_checkpoints.py --output-dir ./checkpoints/resnet_sin

# Or download a specific checkpoint
python download_resnet_checkpoints.py --checkpoint resnet50_sin
```

### Step 2: Run Validation Tests

**Test ResNet models only** (validates your calculation):
```bash
python test_shape_bias_validation.py \
    --data-root /path/to/your/data \
    --checkpoint-dir ./checkpoints/resnet_sin \
    --test resnet
```

**Test DeiT loading methods only** (compares torch.hub vs HuggingFace):
```bash
python test_shape_bias_validation.py \
    --data-root /path/to/your/data \
    --test deit
```

**Run all tests**:
```bash
python test_shape_bias_validation.py \
    --data-root /path/to/your/data \
    --checkpoint-dir ./checkpoints/resnet_sin \
    --test both
```

---

## Expected Results

### ResNet-50 Models (from Geirhos et al., ICLR 2019)

| Model | Training Data | Expected Shape Bias | Source |
|-------|---------------|---------------------|--------|
| ResNet-50 (IN) | ImageNet | **~22%** | torchvision pretrained |
| ResNet-50 (SIN) | Stylized-ImageNet | **~81%** | Downloaded checkpoint |
| ResNet-50 (SIN+IN) | Joint SIN+IN | **~50-60%** | Downloaded checkpoint |
| Shape-ResNet | SIN+IN → IN finetune | **~70-75%** | Downloaded checkpoint |

**If your results match these values (±3%), your calculation is correct!**

### DeiT Models - Expected Findings

The test will reveal if discrepancies are due to:

1. **Different model sources**:
   - torch.hub loads from Facebook's original repository
   - HuggingFace/timm may have slight differences in final layer initialization

2. **Preprocessing differences** (MOST LIKELY CAUSE):
   - **HuggingFace DeiT processor uses WRONG normalization**: `mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]`
   - **Correct DeiT normalization**: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` (ImageNet)
   - This preprocessing error can cause significant accuracy drops (5-10%)

---

## Understanding the Test Output

The script will produce comparison tables like this:

```
==================================================================================
SHAPE BIAS COMPARISON
==================================================================================
Model                                    Shape Bias     Expected       Difference
----------------------------------------------------------------------------------
ResNet-50 (ImageNet)                     21.85%         22.00%         -0.15%
ResNet-50 (SIN)                          80.73%         81.00%         -0.27%
==================================================================================

DETAILED BREAKDOWN:
----------------------------------------------------------------------------------

ResNet-50 (ImageNet):
  Shape correct:      280
  Texture correct:   1000
  Both correct:         0
  Neither correct:    120
  Total evaluated:   1400
  Shape bias:        21.88%
```

### Interpreting Results

✅ **Your calculation is CORRECT if**:
- ResNet-50 (IN) shows ~22% shape bias (±3%)
- ResNet-50 (SIN) shows ~81% shape bias (±3%)
- Results are consistent across multiple runs

❌ **Your calculation may be WRONG if**:
- ResNet-50 (IN) shows >30% or <15% shape bias
- ResNet-50 (SIN) shows <70% or >90% shape bias
- Large discrepancies from paper values

---

## Diagnosis Guide

### Scenario 1: ResNet results match paper, but DeiT doesn't

**Cause**: Preprocessing or model loading issue with DeiT

**Solutions**:
1. Check normalization values (see code in `datasets.py`)
2. Verify you're using `BICUBIC` interpolation
3. Confirm resize=256, crop=224 for eval
4. Try loading from torch.hub to compare

### Scenario 2: All results significantly off

**Cause**: Issue with shape bias calculation or cue-conflict dataset

**Solutions**:
1. Verify cue-conflict dataset structure:
   ```
   cue-conflict/
   ├── airplane/
   │   ├── airplane1-bear2.png
   │   ├── airplane1-bicycle3.png
   │   └── ...
   ├── bear/
   └── ...
   ```
2. Check that you're using "mean" aggregation (recommended)
3. Verify you're excluding non-conflict images (shape == texture)
4. Ensure you're only counting "correct" predictions (shape OR texture)

### Scenario 3: DeiT torch.hub vs HuggingFace differ significantly

**Cause**: Normalization preprocessing difference

**Expected behavior**: 
- With WRONG HF normalization: Lower accuracy, different shape bias
- With CORRECT ImageNet normalization: Should be nearly identical

**Your code already fixes this** in `datasets.py`:
```python
# DeiT uses ImageNet normalization, NOT [0.5, 0.5, 0.5]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```

---

## Technical Details

### Cue-Conflict Shape Bias Calculation (5-step process)

Your `cue_conflict_evaluator.py` correctly implements this:

1. **Evaluate on 1,280 images** with conflicting shape and texture cues
2. **Map 1000-class predictions → 16 categories** using mean aggregation
3. **Exclude non-conflict images** where shape == texture
4. **Keep only "correct" predictions** (predicted either shape OR texture)
5. **Compute**: `shape_bias = shape_correct / (shape_correct + texture_correct)`

### DeiT Preprocessing (Critical!)

**Correct preprocessing for DeiT**:
```python
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                        std=[0.229, 0.224, 0.225])    # ImageNet std
])
```

**Incorrect HuggingFace default** (causes 5-10% accuracy drop):
```python
transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
```

---

## Files in This Test Suite

| File | Purpose |
|------|---------|
| `test_shape_bias_validation.py` | Main validation script |
| `download_resnet_checkpoints.py` | Download official ResNet checkpoints |
| `README_VALIDATION_TEST.md` | This file |

---

## Checkpoint Downloads

The official checkpoints are hosted at:
https://github.com/rgeirhos/texture-vs-shape/tree/master/models

Checkpoints available:

1. **resnet50_train_60_epochs-c8e5653e.pth.tar**
   - ResNet-50 trained on SIN only
   - Expected shape bias: ~81%

2. **resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar**
   - ResNet-50 trained jointly on SIN+IN
   - Expected shape bias: ~50-60%

3. **resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar**
   - Shape-ResNet: SIN+IN model fine-tuned on IN
   - Expected shape bias: ~70-75%

---

## Common Issues & Solutions

### Q: "Cue-conflict dataset not found"
**A**: Download from https://github.com/rgeirhos/texture-vs-shape/tree/master/stimuli/style-transfer-preprocessed-512

### Q: "Checkpoint download fails"
**A**: The URLs might have changed. Check the latest links at https://github.com/rgeirhos/texture-vs-shape

### Q: "torch.hub fails to load DeiT"
**A**: Requires internet connection on first run. After caching, works offline.

### Q: "Results vary between runs"
**A**: Small variations (<1%) are normal due to:
- Floating point precision
- Batch order (shouldn't affect results with proper evaluation)

### Q: "My ResNet results match, but DeiT is different"
**A**: This is the most common scenario. Check:
1. Normalization values (most likely cause)
2. Interpolation method (should be BICUBIC)
3. Resize/crop values (256/224)
4. Model source (torch.hub vs HuggingFace)

---

## Next Steps After Validation

### If ResNet tests pass (calculation is correct):

1. **For DeiT discrepancies**: Compare torch.hub vs HuggingFace loading
2. **Document findings**: Note which loading method matches original repos
3. **Update your code**: Ensure you're using correct normalization (you already are!)
4. **Re-run experiments**: With confidence in your methodology

### If ResNet tests fail (calculation has issues):

1. **Review** `cue_conflict_evaluator.py` against the paper
2. **Check** the 16-class mapping in `IMAGENET_TO_16CLASS` 
3. **Verify** aggregation method ("mean" recommended)
4. **Compare** detailed outputs with original code

---

## References

- **Paper**: Geirhos et al., "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness", ICLR 2019
- **Official Repo**: https://github.com/rgeirhos/texture-vs-shape
- **Model Checkpoints**: https://github.com/rgeirhos/texture-vs-shape/tree/master/models
- **Cue-Conflict Data**: https://github.com/rgeirhos/texture-vs-shape/tree/master/stimuli

---

## Support

If you encounter issues:
1. Check the diagnosis guide above
2. Review your `cue_conflict_evaluator.py` implementation
3. Compare your preprocessing with the examples in this suite
4. Verify dataset structure and checkpoint integrity

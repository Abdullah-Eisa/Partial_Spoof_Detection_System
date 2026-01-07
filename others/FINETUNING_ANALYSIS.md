# Feature Extractor Finetuning - Current Implementation Analysis

## Summary
**The current code SAVES a finetuned feature extractor during training but DOES NOT load it during inference.**

---

## Current Behavior

### During Training (train.py)
✅ **Feature Extractor Finetuning IS Enabled:**
- When `save_feature_extractor=True` (config: `system.save_feature_extractor: true`)
- The feature extractor is unfrozen for training
- Specifically, only `final_proj` layer is trainable, others are frozen
- Uses separate optimizer with lower learning rate (0.00005) for feature extractor

**Training Code (train.py, lines 196-202):**
```python
if save_feature_extractor:
    feature_extractor_filename = f"w2v_large_lv_fsh_swbd_cv_{timestamp}.pt"
    feature_extractor_save_path = os.path.join(model_save_path, feature_extractor_filename)
    save_checkpoint(feature_extractor, optimizer, NUM_EPOCHS, feature_extractor_save_path)
```

**Saved File Example:**
```
/root/Partial_Spoof_Detection_System/models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt
```

### During Inference (inference.py)
❌ **Feature Extractor Finetuned Checkpoint is NOT Loaded:**
- Inference creates a fresh feature extractor from the original pretrained checkpoint
- Located: `inference.py`, lines 264-265

**Current Code (inference.py, lines 264-265):**
```python
# Load feature extractor
feature_extractor = FeatureExtractorFactory.create_extractor(config, device)
feature_extractor.eval()
```

**What happens in create_extractor:**
```python
# feature_extractors.py, lines 13-43
# Always loads from config['feature_extractor']['ssl_checkpoint']
# Which points to: "${BASE_DIR}/models/w2v_large_lv_fsh_swbd_cv.pt"
# (the original pretrained model, NOT the finetuned version)
```

---

## The Problem

When using a finetuned feature extractor:
1. ✅ Training saves the finetuned weights to `w2v_large_lv_fsh_swbd_cv_TIMESTAMP.pt`
2. ❌ Inference ignores this file and loads the original pretrained model instead
3. ❌ Results are inconsistent: training uses finetuned weights, inference uses original weights

---

## Solution Required

To properly use the finetuned feature extractor, you need to:

### Option 1: Add config option for finetuned checkpoint path
**Modify default_config.yaml:**
```yaml
feature_extractor:
  type: "wav2vec2"
  ssl_checkpoint: "${BASE_DIR}/models/w2v_large_lv_fsh_swbd_cv.pt"
  # ADD THIS:
  finetuned_checkpoint: null  # or path to finetuned checkpoint
```

### Option 2: Modify inference.py to accept finetuned checkpoint path
**Add parameter to FeatureExtractorFactory.create_extractor()** to optionally load finetuned weights:
```python
feature_extractor = FeatureExtractorFactory.create_extractor(
    config, 
    device,
    finetuned_checkpoint_path="/path/to/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt"
)
```

### Option 3: Store finetuned checkpoint path in training checkpoint
**Modify save_checkpoint()** to include finetuned feature extractor path in the main model checkpoint, then load it in inference.

---

## Current Training Configuration
**config/default_config.yaml, lines 179-181:**
```yaml
system:
  save_feature_extractor: true  # ← Enables finetuning
```

**Optimizer setup during training (model.py, lines 977-982):**
```python
if save_feature_extractor and hasattr(feature_extractor, 'parameters'):
    optimizer = optim.AdamW(
        [{'params': feature_extractor.parameters(), 'lr': 0.00005},
         {'params': PS_Model.parameters()}],  # Different LR for backend
        lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
```

---

## Files to Examine
- [inference.py](inference.py) - Line 264: Feature extractor loading
- [feature_extractors.py](feature_extractors.py) - Lines 13-43: FeatureExtractorFactory
- [model.py](model.py) - Lines 977-982: Optimizer setup for finetuning
- [train.py](train.py) - Lines 196-202: Feature extractor saving
- [config/default_config.yaml](config/default_config.yaml) - Line 181: save_feature_extractor flag

---

## Recommendation
**Add a config option to specify the finetuned feature extractor checkpoint path**, and modify `FeatureExtractorFactory.create_extractor()` to load it if provided. This maintains backward compatibility while enabling proper finetuned model usage.

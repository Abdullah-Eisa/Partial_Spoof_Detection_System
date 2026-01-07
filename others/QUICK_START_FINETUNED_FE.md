# Quick Reference: Finetuned Feature Extractor Setup

## TL;DR - Quick Setup

### To use your finetuned feature extractor in inference:

1. **Locate your finetuned checkpoint** from training:
   ```bash
   ls -la models/back_end_models/w2v_large_lv_fsh_swbd_cv_*.pt
   # Example: w2v_large_lv_fsh_swbd_cv_20260106_130514.pt
   ```

2. **Edit config/default_config.yaml**:
   ```yaml
   feature_extractor:
     type: "wav2vec2"
     ssl_checkpoint: "${BASE_DIR}/models/w2v_large_lv_fsh_swbd_cv.pt"
     # ADD THIS LINE:
     finetuned_checkpoint: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt"
   ```

3. **Run inference**:
   ```bash
   python inference.py
   ```

## Configuration Options

| Setting | Purpose | Default |
|---------|---------|---------|
| `feature_extractor.type` | Which extractor to use | `"wav2vec2"` |
| `feature_extractor.ssl_checkpoint` | Base pretrained model | `"${BASE_DIR}/models/w2v_large_lv_fsh_swbd_cv.pt"` |
| `feature_extractor.finetuned_checkpoint` | **NEW**: Finetuned weights | `null` |

## What Happens

### With `finetuned_checkpoint: null` (Default)
```
Load base model → Use original pretrained weights
```

### With `finetuned_checkpoint: <path>` (Your Case)
```
Load base model → Override with finetuned weights → Use finetuned model
```

## Key Points

✅ **Finetuned checkpoint is optional** - omit it to use original pretrained model  
✅ **Both models must exist** - base checkpoint + finetuned checkpoint (if specified)  
✅ **Training automatically saves finetuned checkpoint** - when `save_feature_extractor: true`  
✅ **Graceful error handling** - if checkpoint missing, falls back to base model  

## Common Scenarios

### Scenario 1: Using Finetuned Model
```yaml
system:
  save_feature_extractor: true  # Enables finetuning during training

feature_extractor:
  finetuned_checkpoint: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt"
```

### Scenario 2: Reset to Original Pretrained
```yaml
feature_extractor:
  finetuned_checkpoint: null  # Use original pretrained model
```

### Scenario 3: Switch Between Models
```yaml
# For ASVspoof2019:
finetuned_checkpoint: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_asvspoof.pt"

# For RFP:
finetuned_checkpoint: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_rfp.pt"
```

## Verify It's Working

When inference starts, you should see:
```
✓ Loaded finetuned feature extractor from .../w2v_large_lv_fsh_swbd_cv_20260106_130514.pt
```

If using original model, you won't see this message.

## Your Current Saved Models

From your workspace:
```
models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt
                       ↑ matches with ↓
ASVspoof2019_LA_Dataset_model_epochs30_batch8_lr0.0006_20260106_130514.pth
```

Use both in config to ensure consistency!

## Files Modified

- ✅ `config/default_config.yaml` - Added `finetuned_checkpoint` option
- ✅ `feature_extractors.py` - Wav2Vec2Extractor and HuBERTExtractor now load finetuned weights
- ✅ `inference.py` - Displays when loading finetuned feature extractor
- ✅ `model.py` - Training also uses finetuned weights if configured

## Need Help?

See: [FINETUNED_FEATURE_EXTRACTOR_GUIDE.md](FINETUNED_FEATURE_EXTRACTOR_GUIDE.md)

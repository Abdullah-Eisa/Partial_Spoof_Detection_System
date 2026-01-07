# Using Finetuned Feature Extractors - Implementation Guide

## Overview
The code now supports loading **finetuned feature extractor checkpoints** during both training and inference. This ensures consistency between training and evaluation phases when `save_feature_extractor: true`.

---

## What Changed

### 1. **Configuration File** (`config/default_config.yaml`)
Added new configuration option:

```yaml
feature_extractor:
  type: "wav2vec2"
  ssl_checkpoint: "${BASE_DIR}/models/w2v_large_lv_fsh_swbd_cv.pt"
  
  # NEW: Path to finetuned feature extractor checkpoint (optional)
  # If save_feature_extractor=true during training, use the saved finetuned checkpoint here
  # Set to null to use the original pretrained model
  finetuned_checkpoint: null
  # Example: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt"
```

### 2. **Feature Extractors** (`feature_extractors.py`)

#### Updated: `FeatureExtractorFactory.create_extractor()`
- Now extracts `finetuned_checkpoint` from config
- Passes it to individual extractor classes

#### Updated: `Wav2Vec2Extractor`
- Added `_load_finetuned_weights()` method
- Loads finetuned weights after base model initialization
- Gracefully handles missing checkpoints with warnings

#### Updated: `HuBERTExtractor`
- Same finetuning support as Wav2Vec2
- Compatible with both pretrained and finetuned models

### 3. **Inference** (`inference.py`)
- Displays message when loading finetuned feature extractor
- Automatically uses `finetuned_checkpoint` if configured

### 4. **Training** (`model.py`)
- Updated `initialize_models()` to log when loading finetuned weights
- Ensures consistency across training sessions

---

## How to Use

### **Step 1: Train with Feature Extractor Finetuning**

Ensure config has:
```yaml
system:
  save_feature_extractor: true
```

During training, the code will save:
1. **Backend model**: `{dataset}_model_epochs{N}_batch{B}_lr{LR}_{timestamp}.pth`
2. **Finetuned feature extractor**: `w2v_large_lv_fsh_swbd_cv_{timestamp}.pt`

Example from your workspace:
```
models/back_end_models/
├── ASVspoof2019_LA_Dataset_model_epochs30_batch8_lr0.0006_20260106_130514.pth
└── w2v_large_lv_fsh_swbd_cv_20260106_130514.pt  ← Finetuned weights (same timestamp)
```

### **Step 2: Use Finetuned Weights in Inference**

Update config to point to the finetuned checkpoint:

```yaml
feature_extractor:
  type: "wav2vec2"
  ssl_checkpoint: "${BASE_DIR}/models/w2v_large_lv_fsh_swbd_cv.pt"
  # Add the finetuned checkpoint from training:
  finetuned_checkpoint: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt"

paths:
  ps_model_checkpoint: "${BASE_DIR}/models/back_end_models/ASVspoof2019_LA_Dataset_model_epochs30_batch8_lr0.0006_20260106_130514.pth"
```

### **Step 3: Run Inference**

```bash
python inference.py
```

The code will now:
1. ✅ Load pretrained model from `ssl_checkpoint`
2. ✅ Load finetuned weights from `finetuned_checkpoint`
3. ✅ Use the finetuned feature extractor for inference

You should see output like:
```
✓ Loaded finetuned feature extractor from .../w2v_large_lv_fsh_swbd_cv_20260106_130514.pt
```

---

## Examples

### Example 1: Using Original Pretrained Model
```yaml
feature_extractor:
  finetuned_checkpoint: null  # or just omit this line
```
Result: Uses original `ssl_checkpoint` model

### Example 2: Using Finetuned Model
```yaml
feature_extractor:
  finetuned_checkpoint: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt"
```
Result: Loads base model, then overrides with finetuned weights

### Example 3: Multiple Finetuned Models
You can maintain different finetuned versions for different datasets:

```yaml
# For ASVspoof2019 training
feature_extractor:
  finetuned_checkpoint: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_asvspoof_20260106.pt"

# For RFP training
feature_extractor:
  finetuned_checkpoint: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_rfp_20260106.pt"
```

---

## Workflow Summary

```
Training Phase:
  ├─ Load pretrained feature extractor
  ├─ Finetune final_proj layer (if save_feature_extractor=true)
  ├─ Train backend model
  └─ Save both:
      ├─ Finetuned feature extractor: w2v_large_..._TIMESTAMP.pt
      └─ Backend model: MODEL_TIMESTAMP.pth

Inference Phase:
  ├─ Load pretrained feature extractor from ssl_checkpoint
  ├─ Load finetuned weights from finetuned_checkpoint (if configured)
  ├─ Load backend model from ps_model_checkpoint
  └─ Run inference with finetuned feature extractor
```

---

## Backward Compatibility

✅ **Fully backward compatible:**
- If `finetuned_checkpoint: null` (default), uses original pretrained model
- Existing configs work without modification
- Optional feature that doesn't break existing workflows

---

## Technical Details

### How Finetuned Weights are Loaded

1. **Base Model Loading** (in Wav2Vec2Extractor):
   ```python
   self.model = torch.hub.load('s3prl/s3prl', 'wav2vec2', 
                               model_path=checkpoint_path).to(device)
   ```

2. **Finetuned Weights Loading** (if checkpoint provided):
   ```python
   if finetuned_checkpoint is not None:
       finetuned_model = torch.hub.load('s3prl/s3prl', 'wav2vec2',
                                       model_path=finetuned_checkpoint).to(device)
       self.model.load_state_dict(finetuned_model.state_dict())
   ```

3. **Error Handling**:
   - If finetuned checkpoint not found → warning, use base model
   - If loading fails → warning, use base model
   - Graceful degradation ensures inference always works

---

## Troubleshooting

### Issue: "Finetuned checkpoint not found"
**Solution**: Check the path in config matches actual file location
```bash
ls -la models/back_end_models/w2v_large_lv_fsh_swbd_cv_*.pt
```

### Issue: Inference using wrong feature extractor
**Solution**: Verify in inference output for confirmation message
```
✓ Loaded finetuned feature extractor from ...
```

### Issue: Want to reset to original pretrained model
**Solution**: Set `finetuned_checkpoint: null` in config

---

## Files Modified

1. **config/default_config.yaml**
   - Added `feature_extractor.finetuned_checkpoint` config option

2. **feature_extractors.py**
   - Modified `FeatureExtractorFactory.create_extractor()` to extract and pass `finetuned_checkpoint`
   - Added `_load_finetuned_weights()` method to `Wav2Vec2Extractor`
   - Added `_load_finetuned_weights()` method to `HuBERTExtractor`

3. **inference.py**
   - Added logging to indicate when loading finetuned weights

4. **model.py**
   - Updated `initialize_models()` to log finetuned checkpoint loading

---

## Benefits

✅ **Consistency**: Training and inference use same feature extractor weights  
✅ **Flexibility**: Easy to switch between pretrained and finetuned versions  
✅ **Traceability**: Timestamps ensure matching model versions  
✅ **Simplicity**: Config-based approach, no code changes needed  
✅ **Robustness**: Graceful error handling for missing files  

---

## Next Steps (Optional)

1. **Automatic Checkpoint Matching**: Store finetuned checkpoint path in main model checkpoint
2. **Model Registry**: Maintain a registry of finetuned models with metadata
3. **Checkpoint Validation**: Add integrity checks for saved checkpoints
4. **Automated Selection**: Auto-detect matching finetuned checkpoints by timestamp

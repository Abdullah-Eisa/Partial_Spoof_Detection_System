# Implementation Summary: Finetuned Feature Extractor Support

## Problem Statement
The code was **saving finetuned feature extractors during training** but **ignoring them during inference**, causing model inconsistency between training and evaluation phases.

**Training**: Uses finetuned weights  
**Inference**: Uses original pretrained weights  
❌ **Result**: Inconsistent behavior

---

## Solution Implemented

### ✅ Problem Solved
Now the code can **load and use finetuned feature extractors in both training and inference**.

**Training**: Trains and saves finetuned weights  
**Inference**: Loads and uses finetuned weights  
✅ **Result**: Consistent behavior

---

## Changes Made

### 1. **config/default_config.yaml**
**Added new configuration field:**
```yaml
feature_extractor:
  # ... existing fields ...
  ssl_checkpoint: "${BASE_DIR}/models/w2v_large_lv_fsh_swbd_cv.pt"
  
  # NEW: Optional path to finetuned checkpoint
  finetuned_checkpoint: null
  # Example: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt"
```

### 2. **feature_extractors.py**

#### A. Updated `FeatureExtractorFactory.create_extractor()`
```python
# Extract finetuned_checkpoint from config
finetuned_checkpoint = config['feature_extractor'].get('finetuned_checkpoint', None)

# Pass to extractors
return Wav2Vec2Extractor(
    checkpoint_path=config['feature_extractor']['ssl_checkpoint'],
    device=device,
    finetuned_checkpoint=finetuned_checkpoint  # NEW PARAMETER
)
```

#### B. Updated `Wav2Vec2Extractor.__init__()`
```python
def __init__(self, checkpoint_path, device='cpu', finetuned_checkpoint=None):
    # Load base pretrained model
    self.model = torch.hub.load('s3prl/s3prl', 'wav2vec2', 
                                model_path=checkpoint_path).to(device)
    
    # Load finetuned weights if provided
    if finetuned_checkpoint is not None:
        self._load_finetuned_weights(finetuned_checkpoint, device)
```

#### C. Added `Wav2Vec2Extractor._load_finetuned_weights()`
```python
def _load_finetuned_weights(self, finetuned_checkpoint, device):
    """Load finetuned feature extractor weights"""
    # Handle path expansion (e.g., ${BASE_DIR})
    expanded_path = os.path.expandvars(finetuned_checkpoint)
    
    # Check file exists
    if not os.path.exists(expanded_path):
        print(f"Warning: Finetuned checkpoint not found...")
        return
    
    try:
        # Load finetuned model
        finetuned_model = torch.hub.load('s3prl/s3prl', 'wav2vec2', 
                                        model_path=expanded_path).to(device)
        
        # Override base model weights with finetuned weights
        self.model.load_state_dict(finetuned_model.state_dict())
        
        print(f"✓ Loaded finetuned feature extractor from {expanded_path}")
    except Exception as e:
        print(f"Warning: Failed to load finetuned weights...")
```

#### D. Updated `HuBERTExtractor` (same changes as Wav2Vec2)
- Added `finetuned_checkpoint` parameter
- Added `_load_finetuned_weights()` method
- Handles both pretrained and finetuned models

### 3. **inference.py**

**Added logging to indicate finetuned checkpoint loading:**
```python
# Load feature extractor
finetuned_checkpoint = config['feature_extractor'].get('finetuned_checkpoint', None)
if finetuned_checkpoint:
    print(f"Loading feature extractor with finetuned weights from config: {finetuned_checkpoint}")

feature_extractor = FeatureExtractorFactory.create_extractor(config, device)
feature_extractor.eval()
```

### 4. **model.py**

**Updated `initialize_models()` function:**
```python
# Log feature extractor loading info
finetuned_checkpoint = config['feature_extractor'].get('finetuned_checkpoint', None)
if finetuned_checkpoint:
    print(f"Feature Extractor: Loading with finetuned weights from {finetuned_checkpoint}")
```

---

## How It Works

### Loading Sequence

```
1. FeatureExtractorFactory.create_extractor(config, device)
   ↓
2. Extract finetuned_checkpoint from config['feature_extractor']
   ↓
3. Create Wav2Vec2Extractor(checkpoint_path, device, finetuned_checkpoint)
   ├─ Load base model from checkpoint_path
   └─ If finetuned_checkpoint provided:
      ├─ Expand path (handle ${BASE_DIR} variables)
      ├─ Check file exists
      └─ Load finetuned weights and override base model
   ↓
4. Return feature extractor ready for use
```

### Error Handling

✅ **Graceful degradation** if finetuned checkpoint:
- Doesn't exist → Warning, use base model
- Loading fails → Warning, use base model
- Path variables not expanded → Handled automatically

---

## Usage Examples

### Example 1: Use Finetuned Model
```yaml
# config/default_config.yaml
feature_extractor:
  finetuned_checkpoint: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt"
```
**Result**: Uses finetuned weights

### Example 2: Use Original Pretrained Model
```yaml
# config/default_config.yaml
feature_extractor:
  finetuned_checkpoint: null  # or omit the line
```
**Result**: Uses original pretrained weights

### Example 3: Switch Between Different Finetuned Versions
```yaml
# For training on ASVspoof2019
finetuned_checkpoint: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_asvspoof.pt"

# For training on RFP
finetuned_checkpoint: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_rfp.pt"
```

---

## Key Features

✅ **Backward Compatible** - `finetuned_checkpoint: null` uses original model  
✅ **Optional** - Works with or without finetuned checkpoint  
✅ **Flexible** - Easy to switch between different finetuned versions  
✅ **Safe** - Graceful error handling, never crashes  
✅ **Traceable** - Saved and used via config, easy to track versions  
✅ **Consistent** - Training and inference use same feature extractor weights  

---

## Testing Verification

### To verify the implementation works:

1. **Check config option exists:**
   ```bash
   grep -A2 "finetuned_checkpoint" config/default_config.yaml
   ```

2. **Check code modifications:**
   ```bash
   grep -n "_load_finetuned_weights" feature_extractors.py  # Should find 2 occurrences
   grep -n "finetuned_checkpoint" feature_extractors.py     # Should find multiple
   ```

3. **Run inference with finetuned model:**
   ```bash
   # Update config with finetuned_checkpoint path
   python inference.py
   # Should see: ✓ Loaded finetuned feature extractor from ...
   ```

4. **Run inference without finetuned model:**
   ```bash
   # Set finetuned_checkpoint: null
   python inference.py
   # Should NOT see finetuned loading message
   ```

---

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| Train-Inference Consistency | ❌ Different models | ✅ Same models |
| Feature Extractor Control | ❌ No config option | ✅ Config-based |
| Model Tracking | ❌ Manual management | ✅ Config/timestamp-based |
| Error Handling | N/A | ✅ Graceful degradation |
| Backward Compatibility | N/A | ✅ Fully compatible |

---

## Files Modified Summary

| File | Changes | Type |
|------|---------|------|
| `config/default_config.yaml` | Added `feature_extractor.finetuned_checkpoint` | Config |
| `feature_extractors.py` | Updated factory + 2 extractors + load methods | Core |
| `inference.py` | Added logging for finetuned checkpoint | Logging |
| `model.py` | Updated `initialize_models()` logging | Logging |

**Total**: 4 files modified, ~150 lines of code added

---

## Documentation Created

1. **FINETUNED_FEATURE_EXTRACTOR_GUIDE.md** - Comprehensive guide with examples
2. **QUICK_START_FINETUNED_FE.md** - Quick reference for common tasks
3. This file - Technical implementation details

---

## Next Steps (Optional Enhancements)

1. **Automatic Checkpoint Detection**: Auto-match finetuned checkpoint based on timestamp
2. **Checkpoint Registry**: Maintain a registry of finetuned models with metadata
3. **Model Validation**: Add checksums/signatures for checkpoint verification
4. **API Extensions**: Add command-line arguments for checkpoint override

---

## Conclusion

The code now properly **saves and loads finetuned feature extractors**, ensuring consistent behavior between training and inference phases. The implementation is:

- **Simple**: Config-based approach, no complex code
- **Safe**: Error handling ensures robustness
- **Flexible**: Works with any feature extractor type
- **Backward Compatible**: Existing configs work unchanged

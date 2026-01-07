# NEXT STEPS: Using Your Finetuned Feature Extractor

## What You Have Now

The code has been updated to **load finetuned feature extractors**. Here's what to do:

---

## Step 1: Identify Your Finetuned Checkpoint

Your saved finetuned feature extractor from training:
```
w2v_large_lv_fsh_swbd_cv_20260106_130514.pt
```

Matches with backend model:
```
ASVspoof2019_LA_Dataset_model_epochs30_batch8_lr0.0006_20260106_130514.pth
```

---

## Step 2: Update Your Config File

Edit: `config/default_config.yaml`

Find this section:
```yaml
feature_extractor:
  type: "wav2vec2"
  ssl_checkpoint: "${BASE_DIR}/models/w2v_large_lv_fsh_swbd_cv.pt"
  
  # Path to finetuned feature extractor checkpoint (optional)
  # If save_feature_extractor=true during training, use the saved finetuned checkpoint here
  # Set to null to use the original pretrained model
  finetuned_checkpoint: null
  # Example: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt"
```

Replace `null` with your checkpoint:
```yaml
feature_extractor:
  type: "wav2vec2"
  ssl_checkpoint: "${BASE_DIR}/models/w2v_large_lv_fsh_swbd_cv.pt"
  
  finetuned_checkpoint: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt"
```

Also ensure your backend model path matches:
```yaml
paths:
  ps_model_checkpoint: "${BASE_DIR}/models/back_end_models/ASVspoof2019_LA_Dataset_model_epochs30_batch8_lr0.0006_20260106_130514.pth"
```

---

## Step 3: Run Inference

```bash
python inference.py
```

You should see output like:
```
Loading feature extractor with finetuned weights from config: ${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt
✓ Loaded finetuned feature extractor from .../w2v_large_lv_fsh_swbd_cv_20260106_130514.pt
Starting inference...
```

---

## Step 4: Verify Results

✅ **Success indicators:**
- See "✓ Loaded finetuned feature extractor" message
- Inference completes without errors
- Results reflect finetuned model performance

❌ **If not working:**
- Check file path is correct: `ls models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt`
- Check config path expansion: Path should start with `${BASE_DIR}` and be in `feature_extractor` section
- Set `finetuned_checkpoint: null` temporarily to verify base model works

---

## Configuration Reference

### Option 1: Use Finetuned Model (Recommended for your case)
```yaml
feature_extractor:
  finetuned_checkpoint: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt"
```
**Result**: Uses finetuned weights trained on your data

### Option 2: Use Original Pretrained Model
```yaml
feature_extractor:
  finetuned_checkpoint: null
```
**Result**: Uses original pretrained weights (baseline)

### Option 3: Switch Between Models
Keep multiple checkpoint paths and uncomment the one you want:
```yaml
feature_extractor:
  # Version 1: Finetuned model
  finetuned_checkpoint: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt"
  
  # Version 2: Original pretrained
  # finetuned_checkpoint: null
```

---

## What Changed in the Code

### Four files were modified to enable this:

1. **config/default_config.yaml**
   - Added `feature_extractor.finetuned_checkpoint` option

2. **feature_extractors.py**
   - `Wav2Vec2Extractor` now loads finetuned weights
   - `HuBERTExtractor` now loads finetuned weights
   - Graceful error handling for missing files

3. **inference.py**
   - Logs when finetuned weights are loaded

4. **model.py**
   - Logs finetuned weights during training

**All changes are backward compatible** - existing configs still work unchanged.

---

## Documentation

For more details, read:
- `QUICK_START_FINETUNED_FE.md` - Quick reference
- `FINETUNED_FEATURE_EXTRACTOR_GUIDE.md` - Comprehensive guide
- `CODE_CHANGES_BEFORE_AFTER.md` - Detailed code changes
- `IMPLEMENTATION_SUMMARY_FINETUNED_FE.md` - Technical details

---

## Summary

| Task | Status | How To |
|------|--------|--------|
| Code updated | ✅ | Done |
| Config option added | ✅ | Done |
| Your finetuned checkpoint saved | ✅ | From training |
| Ready to use | ⏳ | Update config below |

---

## ⚡ TL;DR - Just Do This

1. Open `config/default_config.yaml`
2. Find `finetuned_checkpoint: null` line
3. Change to: `finetuned_checkpoint: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt"`
4. Save file
5. Run: `python inference.py`
6. Done! ✅

---

## Questions?

If something's not working:

1. **Check the file exists**: 
   ```bash
   ls -la models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt
   ```

2. **Check config syntax**: Make sure path is in quotes and correct

3. **Test with original model first**:
   ```yaml
   finetuned_checkpoint: null
   ```
   Run inference to confirm base setup works

4. **Check log messages**: Look for warnings about missing files

5. **Refer to documentation**: See guides listed above

---

**You're all set! Your finetuned feature extractor is ready to be used in inference.** 🚀

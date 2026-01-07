# Code Changes Summary: Before vs After

## File 1: config/default_config.yaml

### BEFORE
```yaml
# Feature Extractor configuration
feature_extractor:
  type: "wav2vec2"  # Options: "wav2vec2", "hubert", "mfcc", "lfcc"
  # type: "hubert"
  # type: "mfcc"
  # type: "lfcc"

  # SSL models (wav2vec2, hubert)
  ssl_checkpoint: "${BASE_DIR}/models/w2v_large_lv_fsh_swbd_cv.pt"
  # ssl_checkpoint: "${BASE_DIR}/models/hubert-large-ls960-ft"
```

### AFTER
```yaml
# Feature Extractor configuration
feature_extractor:
  type: "wav2vec2"  # Options: "wav2vec2", "hubert", "mfcc", "lfcc"
  # type: "hubert"
  # type: "mfcc"
  # type: "lfcc"

  # SSL models (wav2vec2, hubert)
  ssl_checkpoint: "${BASE_DIR}/models/w2v_large_lv_fsh_swbd_cv.pt"
  # ssl_checkpoint: "${BASE_DIR}/models/hubert-large-ls960-ft"
  
  # Path to finetuned feature extractor checkpoint (optional)
  # If save_feature_extractor=true during training, use the saved finetuned checkpoint here
  # Set to null to use the original pretrained model
  finetuned_checkpoint: null
  # Example: "${BASE_DIR}/models/back_end_models/w2v_large_lv_fsh_swbd_cv_20260106_130514.pt"
```

**Changes**: Added `finetuned_checkpoint` configuration option (3 new lines + comments)

---

## File 2: feature_extractors.py

### Change 1: FeatureExtractorFactory.create_extractor()

#### BEFORE
```python
@staticmethod
def create_extractor(config, device='cpu'):
    """
    Create feature extractor based on config
    
    Args:
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Feature extractor instance
    """
    extractor_type = config['feature_extractor']['type'].lower()
    
    if extractor_type == 'wav2vec2':
        return Wav2Vec2Extractor(
            checkpoint_path=config['feature_extractor']['ssl_checkpoint'],
            device=device
        )
    elif extractor_type == 'hubert':
        return HuBERTExtractor(
            checkpoint_path=config['feature_extractor']['ssl_checkpoint'],
            device=device
        )
    # ... rest of extractors
```

#### AFTER
```python
@staticmethod
def create_extractor(config, device='cpu'):
    """
    Create feature extractor based on config
    
    Args:
        config: Configuration dictionary
        device: Device to load model on
        
    Returns:
        Feature extractor instance
    """
    extractor_type = config['feature_extractor']['type'].lower()
    
    # Get finetuned checkpoint path if available
    finetuned_checkpoint = config['feature_extractor'].get('finetuned_checkpoint', None)
    
    if extractor_type == 'wav2vec2':
        return Wav2Vec2Extractor(
            checkpoint_path=config['feature_extractor']['ssl_checkpoint'],
            device=device,
            finetuned_checkpoint=finetuned_checkpoint
        )
    elif extractor_type == 'hubert':
        return HuBERTExtractor(
            checkpoint_path=config['feature_extractor']['ssl_checkpoint'],
            device=device,
            finetuned_checkpoint=finetuned_checkpoint
        )
    # ... rest of extractors
```

**Changes**: Extract finetuned_checkpoint and pass to extractors (3 new lines)

---

### Change 2: Wav2Vec2Extractor

#### BEFORE
```python
class Wav2Vec2Extractor(nn.Module):
    """Wav2Vec 2.0 feature extractor"""
    
    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__()
        self.device = device
        self.model = torch.hub.load('s3prl/s3prl', 'wav2vec2', 
                                    model_path=checkpoint_path).to(device)
        self.model.eval()
```

#### AFTER
```python
class Wav2Vec2Extractor(nn.Module):
    """Wav2Vec 2.0 feature extractor"""
    
    def __init__(self, checkpoint_path, device='cpu', finetuned_checkpoint=None):
        super().__init__()
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.finetuned_checkpoint = finetuned_checkpoint
        
        # Load base pretrained model
        self.model = torch.hub.load('s3prl/s3prl', 'wav2vec2', 
                                    model_path=checkpoint_path).to(device)
        self.model.eval()
        
        # Load finetuned weights if provided
        if finetuned_checkpoint is not None:
            self._load_finetuned_weights(finetuned_checkpoint, device)
```

**Changes**: Added finetuned_checkpoint parameter and loading logic (7 new lines)

---

### Change 3: Wav2Vec2Extractor._load_finetuned_weights()

#### NEW METHOD (didn't exist before)
```python
def _load_finetuned_weights(self, finetuned_checkpoint, device):
    """Load finetuned feature extractor weights"""
    import os
    
    # Expand path variables if needed
    expanded_path = os.path.expandvars(finetuned_checkpoint)
    
    if not os.path.exists(expanded_path):
        print(f"Warning: Finetuned checkpoint not found at {expanded_path}")
        print(f"Using base pretrained model from {self.checkpoint_path}")
        return
    
    try:
        # Load the finetuned model
        finetuned_model = torch.hub.load('s3prl/s3prl', 'wav2vec2', 
                                        model_path=expanded_path).to(device)
        
        # Copy the state dict from finetuned model to this model
        self.model.load_state_dict(finetuned_model.state_dict())
        
        print(f"✓ Loaded finetuned feature extractor from {expanded_path}")
    except Exception as e:
        print(f"Warning: Failed to load finetuned weights from {expanded_path}")
        print(f"Error: {str(e)}")
        print(f"Using base pretrained model from {self.checkpoint_path}")
```

**Changes**: New method added (26 lines)

---

### Change 4: HuBERTExtractor

#### Similar changes as Wav2Vec2Extractor:
- Added `finetuned_checkpoint` parameter to `__init__()`
- Added `_load_finetuned_weights()` method
- Same error handling logic

**Changes**: Same as Wav2Vec2 (~30 lines total)

---

## File 3: inference.py

### Change: Load finetuned feature extractor

#### BEFORE
```python
# Load feature extractor
feature_extractor = FeatureExtractorFactory.create_extractor(config, device)
feature_extractor.eval()

# Get feature dimension
from feature_extractors import get_feature_dim_from_config
feature_dim = get_feature_dim_from_config(config)
```

#### AFTER
```python
# Load feature extractor
finetuned_checkpoint = config['feature_extractor'].get('finetuned_checkpoint', None)
if finetuned_checkpoint:
    print(f"Loading feature extractor with finetuned weights from config: {finetuned_checkpoint}")

feature_extractor = FeatureExtractorFactory.create_extractor(config, device)
feature_extractor.eval()

# Get feature dimension
from feature_extractors import get_feature_dim_from_config
feature_dim = get_feature_dim_from_config(config)
```

**Changes**: Added logging for finetuned checkpoint loading (3 new lines)

---

## File 4: model.py

### Change: Log finetuned checkpoint loading in initialize_models()

#### BEFORE
```python
def initialize_models(config, save_feature_extractor=False, LEARNING_RATE=0.0001, DEVICE='cpu'):
    """Initialize the model, feature extractor, and optimizer with sequence model support"""
    from feature_extractors import FeatureExtractorFactory
    
    # Create feature extractor based on config
    feature_extractor = FeatureExtractorFactory.create_extractor(config, DEVICE)
    
    # Get base feature dimension from config
    base_feature_dim = feature_extractor.get_feature_dim()
    
    # Get pooling strategy and sequence model type from config
    pooling_strategy = config['model'].get('pooling_strategy', 'self_weighted')
    sequence_model_type = config['model'].get('sequence_model_type', 'conformer')
    sequence_model_config = config['model'].get('sequence_model_config', None)
    
    print(f"Feature Extractor Type: {config['feature_extractor']['type']}")
    print(f"Base Feature Dim: {base_feature_dim}")
```

#### AFTER
```python
def initialize_models(config, save_feature_extractor=False, LEARNING_RATE=0.0001, DEVICE='cpu'):
    """Initialize the model, feature extractor, and optimizer with sequence model support"""
    from feature_extractors import FeatureExtractorFactory
    
    # Create feature extractor based on config
    feature_extractor = FeatureExtractorFactory.create_extractor(config, DEVICE)
    
    # Get base feature dimension from config
    base_feature_dim = feature_extractor.get_feature_dim()
    
    # Get pooling strategy and sequence model type from config
    pooling_strategy = config['model'].get('pooling_strategy', 'self_weighted')
    sequence_model_type = config['model'].get('sequence_model_type', 'conformer')
    sequence_model_config = config['model'].get('sequence_model_config', None)
    
    # Log feature extractor loading info
    finetuned_checkpoint = config['feature_extractor'].get('finetuned_checkpoint', None)
    if finetuned_checkpoint:
        print(f"Feature Extractor: Loading with finetuned weights from {finetuned_checkpoint}")
    
    print(f"Feature Extractor Type: {config['feature_extractor']['type']}")
    print(f"Base Feature Dim: {base_feature_dim}")
```

**Changes**: Added logging for finetuned checkpoint (4 new lines)

---

## Summary of Changes

| File | Type | Lines Added | Description |
|------|------|-------------|-------------|
| `config/default_config.yaml` | Config | 4 | New `finetuned_checkpoint` option |
| `feature_extractors.py` | Core | 65 | Factory update + 2 extractors + 2 load methods |
| `inference.py` | Logging | 3 | Log finetuned checkpoint loading |
| `model.py` | Logging | 4 | Log finetuned checkpoint in training |
| **TOTAL** | | **76** | Core logic + logging + config |

---

## Backward Compatibility

✅ **All changes are backward compatible:**

- `finetuned_checkpoint: null` is default (optional)
- Existing configs without this field work unchanged
- Feature extractors work with or without finetuned checkpoint
- Graceful error handling for missing files
- No breaking changes to function signatures

---

## Impact on Workflow

### Training Workflow
1. Feature extractor finetuning: **UNCHANGED**
2. Saving finetuned weights: **UNCHANGED**
3. New: Can reload finetuned weights in next training run

### Inference Workflow
1. Before: Loaded original pretrained model ❌
2. After: Can load finetuned model from config ✅
3. New logging: Shows which model is being used ✅

### Key Improvement
**Before**: `Training uses finetuned weights` → `Inference uses original weights` → **INCONSISTENT**  
**After**: `Training uses finetuned weights` → `Inference uses finetuned weights` → **CONSISTENT** ✅

# Pooling Strategies Documentation

This document describes the new pooling strategies implemented in the Partial Spoof Detection System.

## Overview

The model now supports multiple pooling strategies for downsampling and aggregating features:

1. **Self-Weighted Pooling** (Default) - Attention-based pooling using learnable weights
2. **Max Pooling** - Standard maximum pooling operation
3. **Average Pooling** - Standard average pooling with configurable kernel size and stride
4. **Attention Pooling** - Learned feature projection with global pooling
5. **Strided Convolution Pooling** - Downsampling using strided convolutions

## Configuration

Pooling strategies are configured in `config/default_config.yaml` under the `model` section:

```yaml
model:
  pooling_strategy: "self_weighted"  # Options: "self_weighted", "max", "average", "attention", "strided_conv"
  
  # Average pooling parameters
  average_pooling:
    kernel_size: 3
    stride: 3
  
  # Attention pooling (LearnedFeatureProjection) parameters
  attention_pooling:
    output_dim: 256  # Output dimension after attention projection
  
  # Strided convolution pooling parameters
  strided_conv_pooling:
    out_channels: 256
    kernel_size: 3
    stride: 3
    padding: 0
```

## Strategy Details

### 1. Self-Weighted Pooling (Default)

**Class:** `SelfWeightedPooling` (inherited from existing implementation)

**Description:** 
Attention-based pooling that learns weights for each feature dimension. The model learns what features are most important for the classification task.

**Input/Output:**
- Input: (batch, time, feature_dim)
- Output: (batch, feature_dim)

**Configuration:** No specific parameters needed

**Use Case:** General-purpose pooling with learnable attention weights

### 2. Max Pooling

**Class:** `nn.MaxPool1d` (PyTorch built-in)

**Description:**
Standard maximum pooling operation. Takes the maximum value over a sliding window.

**Input/Output:**
- Input: (batch, time, feature_dim)
- Output: (batch, downsampled_time, feature_dim)

**Configuration:**
```yaml
max_pooling_factor: 3  # Kernel size and stride
use_max_pooling: true  # Enable/disable max pooling
```

**Use Case:** Simple downsampling when maximum values are important

### 3. Average Pooling

**Class:** `AveragePooling`

**Description:**
Computes the average value over a sliding window. Reduces temporal dimension while preserving feature dimension.

**Input/Output:**
- Input: (batch, time, feature_dim)
- Output: (batch, downsampled_time, feature_dim)

**Configuration:**
```yaml
average_pooling:
  kernel_size: 3      # Size of the pooling window
  stride: 3           # Stride between windows
```

**Use Case:** Smooth downsampling that preserves all feature information

### 4. Attention Pooling (LearnedFeatureProjection)

**Class:** `LearnedFeatureProjection`

**Description:**
Projects input features to a lower-dimensional space using learned attention weights. Each output feature is a weighted combination of all input features. Includes global pooling (mean) across time dimension.

**Input/Output:**
- Input: (batch, time, input_dim)
- Processing: Attention projection → (batch, time, output_dim)
- Output: (batch, output_dim) [after global mean pooling]

**Configuration:**
```yaml
attention_pooling:
  output_dim: 256     # Target dimension for projection
```

**Use Case:** Dimensionality reduction with learned feature relationships

**Mathematical Operation:**
```
attention = softmax(W, dim=1)  # (output_dim, input_dim)
projected = x @ attention.T    # (batch, time, output_dim)
output = mean(projected, dim=1) # (batch, output_dim)
```

### 5. Strided Convolution Pooling

**Class:** `StridedConvPooling`

**Description:**
Uses 1D convolution with stride > 1 for downsampling. Can learn more complex feature transformations than simple pooling.

**Input/Output:**
- Input: (batch, time, in_channels)
- Output: (batch, downsampled_time, out_channels)

**Configuration:**
```yaml
strided_conv_pooling:
  out_channels: 256   # Number of output channels
  kernel_size: 3      # Convolution kernel size
  stride: 3           # Stride for downsampling
  padding: 0          # Padding (typically 0 for downsampling)
```

**Use Case:** Learnable downsampling with feature transformation

## Usage Examples

### Example 1: Using Average Pooling

```yaml
# config/default_config.yaml
model:
  pooling_strategy: "average"
  average_pooling:
    kernel_size: 3
    stride: 3
```

### Example 2: Using Attention Pooling

```yaml
# config/default_config.yaml
model:
  pooling_strategy: "attention"
  attention_pooling:
    output_dim: 256
```

### Example 3: Using Strided Conv Pooling

```yaml
# config/default_config.yaml
model:
  pooling_strategy: "strided_conv"
  strided_conv_pooling:
    out_channels: 256
    kernel_size: 3
    stride: 3
    padding: 0
```

## Model Architecture Flow

```
Input Features (batch, time, feature_dim)
    ↓
[Pooling Strategy Applied]
    ├─ self_weighted: Direct to Conformer (no downsampling)
    ├─ max: MaxPool1d downsampling
    ├─ average: AveragePooling downsampling
    ├─ attention: LearnedFeatureProjection + global pooling
    └─ strided_conv: StridedConvPooling downsampling
    ↓
Conformer Model
    ↓
Self-Weighted Pooling (global aggregation)
    ↓
Feed-forward Refinement
    ↓
Output Score (batch, 1)
```

## Selecting the Right Strategy

| Strategy | Computational Cost | Memory Usage | Feature Preservation | Best For |
|----------|------------------|--------------|-------------------|----|
| Self-Weighted | Medium | Medium | High | Default, balanced approach |
| Max | Low | Low | Medium | When peak values matter |
| Average | Low | Low | High | Smooth downsampling |
| Attention | Medium-High | Medium | High | Learned dimensionality reduction |
| Strided Conv | Medium-High | Medium | High | Learnable feature transformation |

## Training with Different Strategies

The training pipeline automatically handles different pooling strategies. No changes to training code are needed:

```python
# train.py will use the pooling strategy from config
from train import train_model

# Configure pooling strategy in config/default_config.yaml
# Then run training normally
if __name__ == '__main__':
    train(config)
```

## Performance Considerations

1. **Self-Weighted Pooling**: Baseline approach with learnable attention
2. **Average Pooling**: Fastest, good for quick prototyping
3. **Max Pooling**: Fast but may lose subtle variations
4. **Attention Pooling**: Dimensionality reduction helps with larger models
5. **Strided Conv**: Most flexible but slightly slower due to learnable parameters

## Troubleshooting

### Issue: Dimension mismatch errors
**Solution:** Ensure `num_heads` divides the feature dimension after pooling. The model will automatically adjust dimensions to be divisible by `num_heads`.

### Issue: Out of memory with attention/strided conv
**Solution:** Reduce `output_dim` for attention pooling or `out_channels` for strided conv pooling.

### Issue: Training not converging
**Solution:** Try different pooling strategies. Attention pooling and strided conv provide learnable parameters that might help with complex datasets.

## Testing

Run the test suite to verify pooling strategies:

```bash
python3 test_pooling_strategies.py
```

This tests all pooling strategies and verifies:
- Correct output shapes
- Forward pass compatibility
- Model initialization with each strategy

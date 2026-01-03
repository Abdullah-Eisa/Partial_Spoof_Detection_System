# Sequence Models Guide

This guide shows how to use different sequence model architectures (Conformer, LSTM, Transformer, CNN, TCN) in the Partial Spoof Detection System.

## Available Sequence Models

### 1. **Conformer** (Default)
- **Best for**: Balanced accuracy and efficiency (combines self-attention + convolution)
- **Output dim**: Maintains input dimension
- **Key features**: Combines multi-headed self-attention with depthwise convolution

```yaml
sequence_model_type: 'conformer'
sequence_model_config:
  num_heads: 8
  hidden_dim: 256
  num_layers: 4
  depthwise_conv_kernel_size: 31
  dropout: 0.2
```

### 2. **LSTM**
- **Best for**: Sequence modeling with bidirectional context (RNN-based)
- **Output dim**: 2 × hidden_dim (bidirectional)
- **Key features**: Handles long-range dependencies with gating mechanism

```yaml
sequence_model_type: 'lstm'
sequence_model_config:
  hidden_dim: 128
  num_layers: 2
  dropout: 0.2
```

### 3. **Transformer**
- **Best for**: Long-range dependencies and parallel processing (attention-only)
- **Output dim**: input_dim (maintains dimension)
- **Key features**: Pure attention-based, scales well with sequence length

```yaml
sequence_model_type: 'transformer'
sequence_model_config:
  num_heads: 8
  hidden_dim: 256
  num_layers: 4
  dropout: 0.2
```

### 4. **CNN**
- **Best for**: Local feature extraction and fast inference (convolutional)
- **Output dim**: hidden_dim
- **Key features**: Efficient local context capture with residual connections

```yaml
sequence_model_type: 'cnn'
sequence_model_config:
  hidden_dim: 256
  num_layers: 4
  kernel_size: 3
  dropout: 0.2
```

### 5. **TCN** (Temporal Convolutional Network)
- **Best for**: Sequential processing with dilated convolutions (advanced CNN)
- **Output dim**: hidden_dim
- **Key features**: Dilated convolutions provide exponential receptive field

```yaml
sequence_model_type: 'tcn'
sequence_model_config:
  hidden_dim: 256
  num_layers: 4
  kernel_size: 3
  dropout: 0.2
```

## Configuration

Edit `/root/Partial_Spoof_Detection_System/config/default_config.yaml`:

```yaml
model:
  sequence_model_type: 'conformer'  # Change to 'lstm', 'transformer', 'cnn', or 'tcn'
  sequence_model_config: {}          # Leave empty for defaults, or override specific params
```

## Training with Different Models

### Example 1: Train with LSTM

```yaml
model:
  sequence_model_type: 'lstm'
  sequence_model_config:
    hidden_dim: 128
    num_layers: 2
    dropout: 0.2
```

### Example 2: Train with Transformer

```yaml
model:
  sequence_model_type: 'transformer'
  sequence_model_config:
    num_heads: 8
    hidden_dim: 512
    num_layers: 6
    dropout: 0.2
```

### Example 3: Train with TCN

```yaml
model:
  sequence_model_type: 'tcn'
  sequence_model_config:
    hidden_dim: 256
    num_layers: 5
    kernel_size: 3
    dropout: 0.2
```

## Integration with Pooling Strategies

Sequence models work seamlessly with existing pooling strategies:

```yaml
model:
  pooling_strategy: 'strided_conv'  # or 'self_weighted', 'attention', 'max', 'average'
  sequence_model_type: 'lstm'
  sequence_model_config:
    hidden_dim: 128
    num_layers: 2
```

## Implementation Details

### Files Modified
- **model.py**: BinarySpoofingClassificationModel now supports all sequence model types
- **config/default_config.yaml**: Added sequence_model_type and sequence_model_config fields
- **train.py**: Automatically passes sequence model config to initialization

### Model Selection Logic
The model type is selected in BinarySpoofingClassificationModel.__init__():

```python
if self.sequence_model_type == 'conformer':
    self.sequence_model = tam.Conformer(...)
elif self.sequence_model_type == 'lstm':
    self.sequence_model = LSTMSequenceModel(...)
elif self.sequence_model_type == 'transformer':
    self.sequence_model = TransformerSequenceModel(...)
# ... etc
```

### Output Dimension Handling
Each model's output dimension is automatically detected:
- Conformer: input_dim
- LSTM: 2 × hidden_dim (bidirectional)
- Transformer: input_dim
- CNN: hidden_dim
- TCN: hidden_dim

## Running Training

```bash
# Train with default Conformer
python train.py

# Train with specific sequence model (update config first)
# Edit config/default_config.yaml and set sequence_model_type
python train.py
```

## Model Comparison Recommendations

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|--------|----------|
| Conformer | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Balanced approach |
| LSTM | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Long sequences |
| Transformer | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | Parallel processing |
| CNN | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Real-time inference |
| TCN | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Hierarchical features |

## Troubleshooting

### Issue: "Unknown sequence_model_type"
Make sure sequence_model_type is one of: 'conformer', 'lstm', 'transformer', 'cnn', 'tcn'

### Issue: Dimension mismatch
Sequence models automatically adapt to feature_dim. If using pooling strategies, the sequence model receives pooled dimensions.

### Issue: Out of memory
Try reducing:
- hidden_dim
- num_layers (conformer_layers)
- batch_size in training config

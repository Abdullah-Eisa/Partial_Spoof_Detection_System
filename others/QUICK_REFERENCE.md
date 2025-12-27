# Quick Reference: Pooling Strategies

## Quick Start

Change one line in `config/default_config.yaml`:

```yaml
model:
  pooling_strategy: "attention"  # Change this to switch strategies
```

## Available Strategies

| Strategy | Config Name | Best For | Memory Efficient |
|----------|------------|----------|-----------------|
| Self-Weighted | `self_weighted` | Balanced baseline | ✓ |
| Max | `max` | Peak feature preservation | ✓ |
| Average | `average` | Smooth downsampling | ✓ |
| Attention | `attention` | Learned dim reduction | ✗ |
| Strided Conv | `strided_conv` | Feature learning | ✗ |

## Configuration Snippets

### Self-Weighted (Default)
```yaml
model:
  pooling_strategy: "self_weighted"
```

### Average Pooling
```yaml
model:
  pooling_strategy: "average"
  average_pooling:
    kernel_size: 3
    stride: 3
```

### Attention Pooling
```yaml
model:
  pooling_strategy: "attention"
  attention_pooling:
    output_dim: 256  # Reduce from 768 to 256
```

### Strided Convolution
```yaml
model:
  pooling_strategy: "strided_conv"
  strided_conv_pooling:
    out_channels: 256    # Output feature dimension
    kernel_size: 3       # Convolution window size
    stride: 3           # Downsampling factor
    padding: 0
```

### Max Pooling
```yaml
model:
  pooling_strategy: "max"
  max_pooling_factor: 3    # Downsampling factor
  use_max_pooling: true    # Enable/disable
```

## Implementation Details

### LearnedFeatureProjection (Attention)
```python
# Learned attention weights for feature projection
class LearnedFeatureProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.attention_weights = nn.Parameter(
            torch.Tensor(output_dim, input_dim)
        )
        torch_init.xavier_uniform_(self.attention_weights)
    
    def forward(self, x):
        # x: (batch, time, input_dim)
        attention = F.softmax(self.attention_weights, dim=1)
        projected = torch.matmul(x, attention.t())  # (batch, time, output_dim)
        return torch.mean(projected, dim=1)  # (batch, output_dim)
```

### AveragePooling
```python
class AveragePooling(nn.Module):
    def __init__(self, kernel_size, stride):
        self.pool = nn.AvgPool1d(kernel_size, stride)
    
    def forward(self, x):
        # x: (batch, time, features)
        x = x.transpose(1, 2)  # (batch, features, time)
        x = self.pool(x)
        return x.transpose(1, 2)
```

### StridedConvPooling
```python
class StridedConvPooling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.downsample = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride
        )
    
    def forward(self, x):
        # x: (batch, time, in_channels)
        x = x.transpose(1, 2)
        x = self.downsample(x)  # (batch, out_channels, time')
        return x.transpose(1, 2)
```

## Output Dimensions

| Strategy | Input | Output | Example |
|----------|-------|--------|---------|
| self_weighted | (B, T, 768) | (B, 768) | After global pooling |
| max | (B, T, 768) | (B, T/3, 768) | After max pool factor=3 |
| average | (B, T, 768) | (B, T/3, 768) | After avg pool k=3, s=3 |
| attention | (B, T, 768) | (B, 256) | After projection+global pool |
| strided_conv | (B, T, 768) | (B, T/3, 256) | After conv k=3, s=3 |

## Training

No code changes needed! Just configure and run:

```bash
# Edit config/default_config.yaml to set pooling_strategy
# Then run training normally
python3 train.py

# Or with wandb
python3 main.py
```

## Testing

```bash
# Run comprehensive test suite
python3 test_pooling_strategies.py

# Expected output: All tests passed successfully! ✓
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Dimension mismatch | Ensure output_dim is divisible by num_heads |
| Out of memory | Reduce output_dim for attention or out_channels for strided_conv |
| Training unstable | Try different strategies - attention/strided_conv add learnable parameters |
| Shapes not matching | Check kernel_size, stride, padding calculations |

## References

See also:
- [POOLING_STRATEGIES.md](POOLING_STRATEGIES.md) - Detailed documentation
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Implementation details
- [test_pooling_strategies.py](test_pooling_strategies.py) - Test suite

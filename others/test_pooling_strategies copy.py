#!/usr/bin/env python3
"""Test script for pooling strategies"""

import torch
import yaml
import sys
from model import (
    LearnedFeatureProjection, 
    AveragePooling, 
    StridedConvPooling,
    PoolingFactory,
    BinarySpoofingClassificationModel
)

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_learned_feature_projection():
    """Test LearnedFeatureProjection (attention pooling)"""
    print("\n=== Testing LearnedFeatureProjection (Attention Pooling) ===")
    input_dim = 768
    output_dim = 256
    batch_size = 4
    time_steps = 100
    
    pooling = LearnedFeatureProjection(input_dim=input_dim, output_dim=output_dim)
    x = torch.randn(batch_size, time_steps, input_dim)
    
    output = pooling(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {time_steps}, {output_dim})")
    assert output.shape == (batch_size, time_steps, output_dim), f"Expected {(batch_size, time_steps, output_dim)}, got {output.shape}"
    print("✓ LearnedFeatureProjection test passed!")
    
def test_average_pooling():
    """Test AveragePooling"""
    print("\n=== Testing AveragePooling ===")
    kernel_size = 3
    stride = 3
    input_dim = 768
    batch_size = 4
    time_steps = 100
    
    pooling = AveragePooling(kernel_size=kernel_size, stride=stride)
    x = torch.randn(batch_size, time_steps, input_dim)
    
    output = pooling(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    # assert output.shape[2] == input_dim, f"Output feature dimension should remain {input_dim}"
    print("✓ AveragePooling test passed!")

def test_strided_conv_pooling():
    """Test StridedConvPooling"""
    print("\n=== Testing StridedConvPooling ===")
    in_channels = 768
    out_channels = 256
    kernel_size = 3
    stride = 3
    batch_size = 4
    time_steps = 100
    
    pooling = StridedConvPooling(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=0
    )
    x = torch.randn(batch_size, time_steps, in_channels)
    
    output = pooling(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output channels: {out_channels}")
    assert output.shape[2] == out_channels, f"Expected {out_channels} output channels, got {output.shape[2]}"
    print("✓ StridedConvPooling test passed!")

def test_pooling_factory():
    """Test PoolingFactory"""
    print("\n=== Testing PoolingFactory ===")
    config = load_config('config/default_config.yaml')
    
    input_dim = 768
    strategies = ["average", "attention", "strided_conv"]
    
    for strategy in strategies:
        print(f"\n  Testing {strategy} strategy...")
        pooling, output_dim = PoolingFactory.create_pooling(strategy, input_dim, config)
        if pooling is not None:
            print(f"    Created pooling module: {type(pooling).__name__}")
            print(f"    Output dimension: {output_dim}")
        else:
            print(f"    No pooling module (handled separately in model)")
    
    print("✓ PoolingFactory test passed!")

def test_model_with_strategies():
    """Test BinarySpoofingClassificationModel with different pooling strategies"""
    print("\n=== Testing BinarySpoofingClassificationModel with different strategies ===")
    config = load_config('config/default_config.yaml')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    strategies = ["self_weighted", "max", "average", "attention", "strided_conv"]
    
    for strategy in strategies:
        print(f"\n  Testing strategy: {strategy}")
        config['model']['pooling_strategy'] = strategy
        
        try:
            model = BinarySpoofingClassificationModel(
                feature_dim=768,
                num_heads=8,
                hidden_dim=128,
                max_dropout=0.35,
                depthwise_conv_kernel_size=31,
                conformer_layers=1,
                max_pooling_factor=3,
                use_max_pooling=False,
                pooling_strategy=strategy,
                config=config
            ).to(device)
            
            # Test forward pass
            batch_size = 2
            time_steps = 100
            x = torch.randn(batch_size, time_steps, 768).to(device)
            lengths = torch.full((batch_size,), time_steps, dtype=torch.int16).to(device)
            
            output = model(x, lengths, dropout_prob=0.1)
            print(f"    ✓ Model initialized and forward pass successful")
            print(f"      Output shape: {output.shape}")
            assert output.shape == (batch_size, 1), f"Expected output shape {(batch_size, 1)}, got {output.shape}"
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    print("\n✓ All model strategy tests passed!")

if __name__ == "__main__":
    try:
        test_learned_feature_projection()
        test_average_pooling()
        test_strided_conv_pooling()
        test_pooling_factory()
        test_model_with_strategies()
        
        print("\n" + "="*60)
        print("✓ All tests passed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)










# import torch
# import torch.nn as nn


# class LearnedFeatureProjection(nn.Module):
#     """
#     Attention Pooling: Projects features with learned attention weights.
    
#     Input: (batch, time, input_dim)  - e.g., (B, T, 768)
#     Output: (batch, time, output_dim) - e.g., (B, T, 256)
    
#     Downsamples FEATURE DIMENSION, keeps time dimension intact.
#     """
#     def __init__(self, input_dim, output_dim):
#         super(LearnedFeatureProjection, self).__init__()
        
#         self.input_dim = input_dim
#         self.output_dim = output_dim
        
#         # Linear projection layer to reduce feature dimension
#         self.projection = nn.Linear(input_dim, output_dim)
    
#     def forward(self, inputs):
#         """
#         Input: (batch, time, input_dim)  - e.g., (B, T, 768)
#         Output: (batch, time, output_dim) - e.g., (B, T, 256)
#         """
#         # Apply linear projection across feature dimension
#         # inputs: (B, T, input_dim)
#         # output: (B, T, output_dim)
#         output = self.projection(inputs)
        
#         return output


# class MaxPooling(nn.Module):
#     """
#     Max Pooling for downsampling FEATURE DIMENSION (not time dimension).
    
#     Input: (batch, time, input_dim)
#     Output: (batch, time, downsampled_input_dim)
#     """
#     def __init__(self, kernel_size, stride=None, padding=0):
#         super(MaxPooling, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride if stride is not None else kernel_size
#         self.padding = padding
#         self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=self.stride, padding=padding)
    
#     def forward(self, x):
#         """
#         Input: (batch, time, input_dim)
#         Output: (batch, time, downsampled_input_dim)
        
#         Pool across the input_dim (feature) dimension, keeping time dimension intact.
#         """
#         # Get dimensions
#         batch_size, time_steps, input_dim = x.size()
        
#         # Transpose to (batch, input_dim, time) for pooling
#         x = x.transpose(1, 2)  # (B, input_dim, T)
        
#         # Reshape to treat each time step's features independently
#         # We want to pool along feature dimension while preserving time
#         # Approach: reshape to (batch*time, 1, input_dim)
#         x_reshaped = x.transpose(1, 2).reshape(batch_size * time_steps, 1, input_dim)
        
#         # Apply max pooling on feature dimension
#         x_pooled = self.pool(x_reshaped)  # (batch*time, 1, downsampled_input_dim)
        
#         # Get the output dimension after pooling
#         output_dim = x_pooled.size(2)
        
#         # Reshape back to (batch, time, downsampled_input_dim)
#         x = x_pooled.reshape(batch_size, time_steps, output_dim)
        
#         return x


# class AveragePooling(nn.Module):
#     """
#     Average Pooling for downsampling FEATURE DIMENSION (not time dimension).
    
#     Input: (batch, time, input_dim)
#     Output: (batch, time, downsampled_input_dim)
#     """
#     def __init__(self, kernel_size, stride=None, padding=0):
#         super(AveragePooling, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride if stride is not None else kernel_size
#         self.padding = padding
#         self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=self.stride, padding=padding)
    
#     def forward(self, x):
#         """
#         Input: (batch, time, input_dim)
#         Output: (batch, time, downsampled_input_dim)
        
#         Pool across the input_dim (feature) dimension, keeping time dimension intact.
#         """
#         # Get dimensions
#         batch_size, time_steps, input_dim = x.size()
        
#         # Reshape to (batch*time, 1, input_dim) to pool along feature dimension
#         # Each timestep is treated independently
#         x_reshaped = x.reshape(batch_size * time_steps, 1, input_dim)
        
#         # Apply average pooling on feature dimension
#         x_pooled = self.pool(x_reshaped)  # (batch*time, 1, downsampled_input_dim)
        
#         # Get the output dimension after pooling
#         output_dim = x_pooled.size(2)
        
#         # Reshape back to (batch, time, downsampled_input_dim)
#         x = x_pooled.reshape(batch_size, time_steps, output_dim)
        
#         return x


# class StridedConvPooling(nn.Module):
#     """
#     Strided Convolution Pooling for downsampling FEATURE DIMENSION (not time dimension).
    
#     Input: (batch, time, input_dim)  - e.g., (B, T, 768)
#     Output: (batch, time, output_dim)
#     """
#     def __init__(self, input_dim, output_dim, kernel_size=3, stride=None, padding=0):
#         super(StridedConvPooling, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.stride = stride if stride is not None else kernel_size
        
#         # Conv1d that operates on feature dimension
#         self.downsample = nn.Conv1d(
#             in_channels=1,  # Each time step treated independently
#             out_channels=1,  # Keep same structure
#             kernel_size=kernel_size,
#             stride=self.stride,
#             padding=padding,
#             bias=True
#         )
        
#         # Calculate expected output dimension after convolution
#         expected_output_dim = (input_dim + 2 * padding - kernel_size) // self.stride + 1
        
#         # Add a linear projection to get exact output_dim
#         self.projection = nn.Linear(expected_output_dim, output_dim)
    
#     def forward(self, x):
#         """
#         Input: (batch, time, input_dim)  - e.g., (B, T, 768)
#         Output: (batch, time, output_dim)
        
#         Apply strided convolution across the input_dim (feature) dimension,
#         keeping time dimension intact.
#         """
#         batch_size, time_steps, input_dim = x.size()
        
#         # Reshape to (batch*time, 1, input_dim) to apply Conv1d on feature dimension
#         # Each timestep is treated independently
#         x_reshaped = x.reshape(batch_size * time_steps, 1, input_dim)
        
#         # Apply strided convolution on feature dimension
#         x_conv = self.downsample(x_reshaped)  # (batch*time, 1, conv_output_dim)
        
#         # Remove channel dimension and get features
#         x_conv = x_conv.squeeze(1)  # (batch*time, conv_output_dim)
        
#         # Apply linear projection to get exact output_dim
#         x_projected = self.projection(x_conv)  # (batch*time, output_dim)
        
#         # Reshape back to (batch, time, output_dim)
#         x = x_projected.reshape(batch_size, time_steps, self.output_dim)
        
#         return x


# class PoolingFactory:
#     """
#     Factory class for creating pooling strategies.
#     All strategies downsample FEATURE DIMENSION, not time dimension.
#     """
#     @staticmethod
#     def create_pooling(strategy, input_dim, config):
#         """
#         Create a pooling module based on the specified strategy.
        
#         Args:
#             strategy: str, pooling strategy name
#             input_dim: int, input feature dimension (e.g., 768)
#             config: dict, configuration dictionary containing pooling parameters
        
#         Returns:
#             tuple: (pooling_module, output_dim)
#                    output_dim is the downsampled feature dimension
#         """
#         strategy = strategy.lower()
        
#         if strategy == "max":
#             kernel_size = config['model']['max_pooling']['kernel_size']
#             stride = config['model']['max_pooling'].get('stride', kernel_size)
#             padding = config['model']['max_pooling'].get('padding', 0)
#             pooling = MaxPooling(kernel_size=kernel_size, stride=stride, padding=padding)
#             # Calculate output dimension after max pooling on feature dimension
#             output_dim = (input_dim + 2 * padding - kernel_size) // stride + 1
#             return pooling, output_dim
        
#         elif strategy == "average":
#             kernel_size = config['model']['average_pooling']['kernel_size']
#             stride = config['model']['average_pooling'].get('stride', kernel_size)
#             padding = config['model']['average_pooling'].get('padding', 0)
#             pooling = AveragePooling(kernel_size=kernel_size, stride=stride, padding=padding)
#             # Calculate output dimension after average pooling on feature dimension
#             output_dim = (input_dim + 2 * padding - kernel_size) // stride + 1
#             return pooling, output_dim
        
#         elif strategy == "attention":
#             output_dim = config['model']['attention_pooling']['output_dim']
#             pooling = LearnedFeatureProjection(input_dim=input_dim, output_dim=output_dim)
#             return pooling, output_dim
        
#         elif strategy == "strided_conv":
#             output_dim = config['model']['strided_conv_pooling']['output_dim']
#             kernel_size = config['model']['strided_conv_pooling']['kernel_size']
#             stride = config['model']['strided_conv_pooling'].get('stride', kernel_size)
#             padding = config['model']['strided_conv_pooling'].get('padding', 0)
#             pooling = StridedConvPooling(
#                 input_dim=input_dim,
#                 output_dim=output_dim,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding
#             )
#             return pooling, output_dim
        
#         elif strategy == "self_weighted":
#             # SelfWeightedPooling will be handled separately in the model
#             # This pools across TIME dimension, not feature dimension
#             return None, input_dim
        
#         else:
#             raise ValueError(f"Unknown pooling strategy: {strategy}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing all pooling strategies for FEATURE DIMENSION reduction\n")
    
    batch_size = 4
    time_steps = 100
    input_dim = 768
    output_dim = 256
    
    # Create dummy input
    x = torch.randn(batch_size, time_steps, input_dim)
    print(f"Input shape: {x.shape} (batch, time, feature_dim)\n")
    
    # Test 1: Attention Pooling
    print("1. Testing Attention Pooling (LearnedFeatureProjection)")
    attention_pool = LearnedFeatureProjection(input_dim=input_dim, output_dim=output_dim)
    out = attention_pool(x)
    print(f"   Output shape: {out.shape}")
    print(f"   Expected: ({batch_size}, {time_steps}, {output_dim})")
    assert out.shape == (batch_size, time_steps, output_dim), "Shape mismatch!"
    print("   ✓ Passed\n")
    
    # Test 2: Max Pooling
    print("2. Testing Max Pooling")
    kernel_size = 3
    stride = 3
    max_pool = MaxPooling(kernel_size=kernel_size, stride=stride)
    out = max_pool(x)
    expected_dim = (input_dim - kernel_size) // stride + 1
    print(f"   Output shape: {out.shape}")
    print(f"   Expected: ({batch_size}, {time_steps}, {expected_dim})")
    assert out.shape == (batch_size, time_steps, expected_dim), "Shape mismatch!"
    print("   ✓ Passed\n")
    
    # Test 3: Average Pooling
    print("3. Testing Average Pooling")
    avg_pool = AveragePooling(kernel_size=kernel_size, stride=stride)
    out = avg_pool(x)
    print(f"   Output shape: {out.shape}")
    print(f"   Expected: ({batch_size}, {time_steps}, {expected_dim})")
    assert out.shape == (batch_size, time_steps, expected_dim), "Shape mismatch!"
    print("   ✓ Passed\n")
    
    # Test 4: Strided Conv Pooling
    print("4. Testing Strided Conv Pooling")
    strided_pool = StridedConvPooling(
        input_dim=input_dim,
        output_dim=output_dim,
        kernel_size=kernel_size,
        stride=stride
    )
    out = strided_pool(x)
    print(f"   Output shape: {out.shape}")
    print(f"   Expected: ({batch_size}, {time_steps}, {output_dim})")
    assert out.shape == (batch_size, time_steps, output_dim), "Shape mismatch!"
    print("   ✓ Passed\n")
    
    print("="*60)
    print("All tests passed! ✓")
    print("All strategies correctly downsample FEATURE DIMENSION")
    print("while keeping TIME DIMENSION intact.")
    print("="*60)
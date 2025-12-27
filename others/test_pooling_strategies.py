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
    print(f"Expected output shape: ({batch_size}, {output_dim})")
    assert output.shape == (batch_size, output_dim), f"Expected {(batch_size, output_dim)}, got {output.shape}"
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
    expected_time = (time_steps + stride - 1) // stride
    print(f"Expected time steps after pooling: {expected_time}")
    assert output.shape[2] == input_dim, f"Output feature dimension should remain {input_dim}"
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

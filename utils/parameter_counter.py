"""
Parameter Counting Utilities for PyTorch Models

Location: utils/parameter_counter.py

This module provides functions to count and analyze trainable and non-trainable
parameters in PyTorch models, with detailed breakdowns by module.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
from collections import OrderedDict


def count_parameters(model: nn.Module, 
                     return_dict: bool = False,
                     verbose: bool = True) -> Dict:
    """
    Count trainable and non-trainable parameters in a PyTorch model.
    
    Args:
        model (nn.Module): PyTorch model to analyze
        return_dict (bool): If True, return detailed dictionary with module breakdown
        verbose (bool): If True, print summary statistics
    
    Returns:
        dict: Dictionary containing parameter counts and optionally module breakdown
    
    Example:
        >>> from utils.parameter_counter import count_parameters
        >>> model = BinarySpoofingClassificationModel(...)
        >>> stats = count_parameters(model, verbose=True)
        >>> print(f"Total params: {stats['total_params']:,}")
    """
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count non-trainable parameters
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    # Total parameters
    total_params = trainable_params + non_trainable_params
    
    # Calculate memory usage (assuming float32)
    trainable_memory_mb = (trainable_params * 4) / (1024 ** 2)
    non_trainable_memory_mb = (non_trainable_params * 4) / (1024 ** 2)
    total_memory_mb = (total_params * 4) / (1024 ** 2)
    
    # Create results dictionary
    results = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'trainable_percentage': (trainable_params / total_params * 100) if total_params > 0 else 0,
        'memory_mb': {
            'total': total_memory_mb,
            'trainable': trainable_memory_mb,
            'non_trainable': non_trainable_memory_mb
        }
    }
    
    if verbose:
        print("\n" + "="*70)
        print("MODEL PARAMETER SUMMARY")
        print("="*70)
        print(f"Total Parameters:          {total_params:>15,}")
        print(f"Trainable Parameters:      {trainable_params:>15,} ({results['trainable_percentage']:.2f}%)")
        print(f"Non-trainable Parameters:  {non_trainable_params:>15,}")
        print("-"*70)
        print(f"Total Memory (float32):    {total_memory_mb:>15.2f} MB")
        print(f"Trainable Memory:          {trainable_memory_mb:>15.2f} MB")
        print(f"Non-trainable Memory:      {non_trainable_memory_mb:>15.2f} MB")
        print("="*70 + "\n")
    
    if return_dict:
        # Add detailed module breakdown
        results['module_breakdown'] = get_module_breakdown(model)
    
    return results


def get_module_breakdown(model: nn.Module, 
                         max_depth: int = 2) -> Dict[str, Dict]:
    """
    Get detailed breakdown of parameters by module.
    
    Args:
        model (nn.Module): PyTorch model to analyze
        max_depth (int): Maximum depth for module hierarchy analysis
    
    Returns:
        dict: Dictionary with parameter counts for each module
    """
    module_stats = OrderedDict()
    
    for name, module in model.named_modules():
        if name == '':  # Skip root module
            continue
        
        # Limit depth
        if name.count('.') >= max_depth:
            continue
        
        # Count parameters directly owned by this module
        trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        non_trainable = sum(p.numel() for p in module.parameters(recurse=False) if not p.requires_grad)
        total = trainable + non_trainable
        
        if total > 0:  # Only include modules with parameters
            module_stats[name] = {
                'total': total,
                'trainable': trainable,
                'non_trainable': non_trainable,
                'trainable_pct': (trainable / total * 100) if total > 0 else 0,
                'memory_mb': (total * 4) / (1024 ** 2)
            }
    
    return module_stats


def print_module_breakdown(model: nn.Module, 
                           max_depth: int = 2,
                           min_params: int = 1000,
                           sort_by: str = 'total') -> None:
    """
    Print detailed breakdown of parameters by module in a formatted table.
    
    Args:
        model (nn.Module): PyTorch model to analyze
        max_depth (int): Maximum depth for module hierarchy
        min_params (int): Minimum parameters to include a module in output
        sort_by (str): Sort by 'total', 'trainable', or 'non_trainable'
    """
    breakdown = get_module_breakdown(model, max_depth=max_depth)
    
    # Filter by minimum parameters
    breakdown = {k: v for k, v in breakdown.items() if v['total'] >= min_params}
    
    # Sort
    sorted_breakdown = sorted(
        breakdown.items(), 
        key=lambda x: x[1][sort_by], 
        reverse=True
    )
    
    print("\n" + "="*100)
    print("MODULE PARAMETER BREAKDOWN")
    print("="*100)
    print(f"{'Module Name':<40} {'Total':>12} {'Trainable':>12} {'Non-train':>12} {'Train %':>8} {'Memory (MB)':>12}")
    print("-"*100)
    
    for name, stats in sorted_breakdown:
        print(f"{name:<40} {stats['total']:>12,} {stats['trainable']:>12,} "
              f"{stats['non_trainable']:>12,} {stats['trainable_pct']:>7.1f}% "
              f"{stats['memory_mb']:>11.2f}")
    
    print("="*100 + "\n")


def quick_param_count(model: nn.Module) -> Tuple[int, int, int]:
    """
    Quick function to get (total, trainable, non_trainable) parameter counts.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        tuple: (total_params, trainable_params, non_trainable_params)
    
    Example:
        >>> from utils.parameter_counter import quick_param_count
        >>> total, trainable, non_trainable = quick_param_count(model)
        >>> print(f"Model has {trainable:,} trainable parameters")
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + non_trainable
    return total, trainable, non_trainable


def get_model_size_mb(model: nn.Module, dtype: str = 'float32') -> float:
    """
    Calculate model size in MB.
    
    Args:
        model (nn.Module): PyTorch model
        dtype (str): Data type ('float32', 'float16', 'int8')
    
    Returns:
        float: Model size in MB
    
    Example:
        >>> from utils.parameter_counter import get_model_size_mb
        >>> size_mb = get_model_size_mb(model, dtype='float32')
        >>> print(f"Model size: {size_mb:.2f} MB")
    """
    bytes_per_param = {'float32': 4, 'float16': 2, 'int8': 1}
    total_params = sum(p.numel() for p in model.parameters())
    size_bytes = total_params * bytes_per_param.get(dtype, 4)
    return size_bytes / (1024 ** 2)


def analyze_gradient_flow(model: nn.Module) -> Dict:
    """
    Analyze which parts of the model have gradients enabled.
    
    Args:
        model (nn.Module): PyTorch model to analyze
    
    Returns:
        dict: Statistics about gradient flow
    
    Example:
        >>> from utils.parameter_counter import analyze_gradient_flow
        >>> grad_stats = analyze_gradient_flow(model)
        >>> print(f"Trainable layers: {grad_stats['trainable_layers']}")
    """
    total_layers = 0
    trainable_layers = 0
    frozen_layers = 0
    
    for name, param in model.named_parameters():
        total_layers += 1
        if param.requires_grad:
            trainable_layers += 1
        else:
            frozen_layers += 1
    
    return {
        'total_layers': total_layers,
        'trainable_layers': trainable_layers,
        'frozen_layers': frozen_layers,
        'trainable_percentage': (trainable_layers / total_layers * 100) if total_layers > 0 else 0
    }


def print_inference_model_info(model: nn.Module, 
                               feature_extractor: Optional[nn.Module] = None,
                               show_breakdown: bool = True) -> Dict:
    """
    Print model information specifically formatted for inference phase.
    
    Args:
        model (nn.Module): Backend classification model
        feature_extractor (nn.Module, optional): Feature extractor model
        show_breakdown (bool): Whether to show detailed module breakdown
    
    Returns:
        dict: Combined statistics
    
    Example:
        >>> from utils.parameter_counter import print_inference_model_info
        >>> stats = print_inference_model_info(PS_Model, feature_extractor)
    """
    print("\n" + "="*80)
    print("INFERENCE - MODEL ARCHITECTURE INFORMATION")
    print("="*80)
    
    # Backend model statistics
    print("\n1. Backend Classification Model:")
    print("-"*80)
    backend_stats = count_parameters(model, verbose=True)
    
    if show_breakdown:
        print_module_breakdown(model, max_depth=2, min_params=1000)
    
    # Feature extractor statistics (if provided)
    combined_stats = {'backend': backend_stats}
    
    if feature_extractor is not None:
        print("\n2. Feature Extractor:")
        print("-"*80)
        fe_stats = count_parameters(feature_extractor, verbose=True)
        combined_stats['feature_extractor'] = fe_stats
        
        # Combined statistics
        print("\n3. Combined System:")
        print("-"*80)
        backend_total, backend_train, _ = quick_param_count(model)
        fe_total, fe_train, _ = quick_param_count(feature_extractor)
        
        combined_total = backend_total + fe_total
        combined_train = backend_train + fe_train
        combined_memory = backend_stats['memory_mb']['total'] + fe_stats['memory_mb']['total']
        
        print(f"Total Parameters:          {combined_total:>15,}")
        print(f"Trainable Parameters:      {combined_train:>15,} ({combined_train/combined_total*100:.2f}%)")
        print(f"Combined Memory (float32): {combined_memory:>15.2f} MB")
        
        combined_stats['combined'] = {
            'total_params': combined_total,
            'trainable_params': combined_train,
            'memory_mb': combined_memory
        }
    
    print("="*80 + "\n")
    
    return combined_stats


    
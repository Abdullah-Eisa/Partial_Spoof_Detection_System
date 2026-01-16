

"""
Statistical Testing Utilities for Model Evaluation
Location: utils/statistical_testing.py
"""

import numpy as np
import torch
from scipy import stats
from typing import Dict, List, Tuple
import json


def bootstrap_confidence_interval(
    predictions: np.ndarray,
    labels: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, float]:
    """
    Calculate bootstrap confidence intervals for a metric.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        metric_fn: Function that computes metric (e.g., EER, accuracy)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with mean, lower_bound, upper_bound
    
    Example:
        >>> from utils.utils import compute_eer
        >>> ci = bootstrap_confidence_interval(preds, labels, 
        ...     metric_fn=lambda p, l: compute_eer(p, l)[0])
        >>> print(f"EER: {ci['mean']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    """
    np.random.seed(random_seed)
    n_samples = len(predictions)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_preds = predictions[indices]
        boot_labels = labels[indices]
        
        # Compute metric on bootstrap sample
        try:
            score = metric_fn(boot_preds, boot_labels)
            bootstrap_scores.append(score)
        except:
            continue
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    return {
        'mean': np.mean(bootstrap_scores),
        'std': np.std(bootstrap_scores),
        'lower': np.percentile(bootstrap_scores, lower_percentile),
        'upper': np.percentile(bootstrap_scores, upper_percentile),
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level
    }


def mcnemar_test(
    predictions_model1: np.ndarray,
    predictions_model2: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Perform McNemar's test for paired model comparison.
    
    Tests null hypothesis: both models have same error rate.
    
    Args:
        predictions_model1: Predictions from first model
        predictions_model2: Predictions from second model
        labels: Ground truth labels
        threshold: Classification threshold
    
    Returns:
        Dictionary with test statistic, p-value, and interpretation
    
    Example:
        >>> result = mcnemar_test(conformer_preds, lstm_preds, labels)
        >>> if result['p_value'] < 0.05:
        ...     print(f"Models significantly different (p={result['p_value']:.4f})")
    """
    # Convert predictions to binary decisions
    pred1_binary = (predictions_model1 >= threshold).astype(int).flatten()
    pred2_binary = (predictions_model2 >= threshold).astype(int).flatten()
    labels_binary = labels.astype(int).flatten()
    
    # Create contingency table
    # correct1_wrong2: Model 1 correct, Model 2 wrong
    # wrong1_correct2: Model 1 wrong, Model 2 correct
    correct1 = (pred1_binary == labels_binary)
    correct2 = (pred2_binary == labels_binary)
    
    correct1_wrong2 = np.sum(correct1 & ~correct2)
    wrong1_correct2 = np.sum(~correct1 & correct2)
    
    # McNemar's test statistic with continuity correction
    n = correct1_wrong2 + wrong1_correct2
    if n == 0:
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'interpretation': 'Models have identical errors'
        }
    
    chi2_stat = ((abs(correct1_wrong2 - wrong1_correct2) - 1) ** 2) / (correct1_wrong2 + wrong1_correct2)
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
    
    return {
        'statistic': chi2_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'correct1_wrong2': int(correct1_wrong2),
        'wrong1_correct2': int(wrong1_correct2),
        'interpretation': 'Significantly different' if p_value < 0.05 else 'Not significantly different'
    }


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict:
    """
    Apply Bonferroni correction for multiple testing.
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise error rate (default: 0.05)
    
    Returns:
        Dictionary with corrected results
    
    Example:
        >>> p_vals = [0.01, 0.03, 0.06, 0.12]
        >>> result = bonferroni_correction(p_vals)
        >>> print(f"Significant tests: {result['n_significant']}/{result['n_tests']}")
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    
    significant = [p < corrected_alpha for p in p_values]
    
    return {
        'n_tests': n_tests,
        'original_alpha': alpha,
        'corrected_alpha': corrected_alpha,
        'p_values': p_values,
        'significant': significant,
        'n_significant': sum(significant),
        'rejected_null': [i for i, sig in enumerate(significant) if sig]
    }


def run_multiple_seeds_experiment(
    config: Dict,
    train_fn,
    inference_fn,
    seeds: List[int] = [42, 123, 456, 789, 1024]
) -> Dict:
    """
    Run experiment with multiple random seeds and aggregate results.
    
    Args:
        config: Configuration dictionary
        train_fn: Training function
        inference_fn: Inference function
        seeds: List of random seeds
    
    Returns:
        Dictionary with aggregated results (mean ± std)
    
    Example:
        >>> from train import train
        >>> from inference import inference
        >>> results = run_multiple_seeds_experiment(config, train, inference)
        >>> print(f"EER: {results['eer_mean']:.4f} ± {results['eer_std']:.4f}")
    """
    results = {
        'eer': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"Running experiment with seed: {seed}")
        print(f"{'='*80}\n")
        
        # Set seed
        import random
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Train model
        train_fn(config)
        
        # Run inference
        metrics = inference_fn(config, show_model_info=False)
        
        # Store results
        results['eer'].append(metrics['utterance_eer'])
        results['precision'].append(metrics.get('precision', 0))
        results['recall'].append(metrics.get('recall', 0))
        results['f1'].append(metrics.get('f1', 0))
    
    # Calculate mean and std
    aggregated = {
        'seeds': seeds,
        'n_runs': len(seeds)
    }
    
    for metric_name, values in results.items():
        aggregated[f'{metric_name}_mean'] = np.mean(values)
        aggregated[f'{metric_name}_std'] = np.std(values)
        aggregated[f'{metric_name}_all'] = values
    
    return aggregated


def save_statistical_results(results: Dict, output_path: str):
    """Save statistical test results to JSON file."""
    import json
    from datetime import datetime
    
    results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Statistical results saved to: {output_path}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Bootstrap confidence intervals
    print("Example: Bootstrap Confidence Intervals")
    print("-" * 60)
    
    # Simulate predictions and labels
    np.random.seed(42)
    predictions = np.random.rand(1000)
    labels = np.random.randint(0, 2, 1000)
    
    # Define metric function (accuracy)
    def accuracy(preds, lbls):
        pred_binary = (preds >= 0.5).astype(int)
        return np.mean(pred_binary == lbls)
    
    ci = bootstrap_confidence_interval(predictions, labels, accuracy)
    print(f"Accuracy: {ci['mean']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    
    # Example: McNemar's test
    print("\nExample: McNemar's Test")
    print("-" * 60)
    
    predictions_model2 = np.random.rand(1000)
    result = mcnemar_test(predictions, predictions_model2, labels)
    print(f"Chi² = {result['statistic']:.4f}, p = {result['p_value']:.4f}")
    print(f"Result: {result['interpretation']}")
    
    # Example: Bonferroni correction
    print("\nExample: Bonferroni Correction")
    print("-" * 60)
    
    p_values = [0.01, 0.03, 0.06, 0.12]
    bonf_result = bonferroni_correction(p_values)
    print(f"Corrected α: {bonf_result['corrected_alpha']:.4f}")
    print(f"Significant tests: {bonf_result['n_significant']}/{bonf_result['n_tests']}")


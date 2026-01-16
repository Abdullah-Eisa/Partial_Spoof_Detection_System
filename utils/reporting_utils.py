
"""
Enhanced Reporting Utilities
Location: utils/reporting_utils.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple
import os


def plot_prediction_distribution(
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: str = 'outputs/prediction_distribution.png',
    bins: int = 50
):
    """
    Plot distribution of prediction scores for genuine vs spoof samples.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        labels: Ground truth labels (0=genuine, 1=spoof)
        save_path: Path to save figure
        bins: Number of histogram bins
    
    Example:
        >>> plot_prediction_distribution(predictions, labels)
    """
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Separate by class
    genuine_preds = predictions[labels == 0]
    spoof_preds = predictions[labels == 1]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot histogram
    axes[0].hist(genuine_preds, bins=bins, alpha=0.6, label='Genuine', color='green', density=True)
    axes[0].hist(spoof_preds, bins=bins, alpha=0.6, label='Spoof', color='red', density=True)
    axes[0].set_xlabel('Prediction Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution of Prediction Scores')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot box plot
    axes[1].boxplot([genuine_preds, spoof_preds], labels=['Genuine', 'Spoof'], 
                    vert=False, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.6))
    axes[1].set_xlabel('Prediction Score')
    axes[1].set_title('Box Plot of Prediction Scores by Class')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Prediction distribution saved to: {save_path}")
    
    # Print statistics
    print("\nPrediction Statistics:")
    print(f"Genuine - Mean: {genuine_preds.mean():.4f}, Std: {genuine_preds.std():.4f}")
    print(f"Spoof   - Mean: {spoof_preds.mean():.4f}, Std: {spoof_preds.std():.4f}")


def find_hardest_samples(
    predictions: np.ndarray,
    labels: np.ndarray,
    file_names: List[str],
    n_samples: int = 20,
    threshold: float = 0.5
) -> Dict:
    """
    Identify hardest samples (lowest confidence correct predictions).
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        file_names: List of file names
        n_samples: Number of hardest samples to return
        threshold: Classification threshold
    
    Returns:
        Dictionary with hardest genuine and spoof samples
    
    Example:
        >>> hardest = find_hardest_samples(preds, labels, file_names)
        >>> print("Hardest genuine samples:")
        >>> for sample in hardest['genuine'][:5]:
        ...     print(f"  {sample['file']}: score={sample['score']:.4f}")
    """
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Find correct predictions
    pred_binary = (predictions >= threshold).astype(int)
    correct_mask = (pred_binary == labels)
    
    # Calculate confidence (distance from threshold)
    confidence = np.abs(predictions - threshold)
    
    # Hardest genuine samples (correct, but low confidence)
    genuine_mask = (labels == 0) & correct_mask
    genuine_indices = np.where(genuine_mask)[0]
    genuine_confidence = confidence[genuine_indices]
    hardest_genuine_idx = genuine_indices[np.argsort(genuine_confidence)[:n_samples]]
    
    # Hardest spoof samples
    spoof_mask = (labels == 1) & correct_mask
    spoof_indices = np.where(spoof_mask)[0]
    spoof_confidence = confidence[spoof_indices]
    hardest_spoof_idx = spoof_indices[np.argsort(spoof_confidence)[:n_samples]]
    
    hardest = {
        'genuine': [
            {
                'file': file_names[i],
                'score': float(predictions[i]),
                'confidence': float(confidence[i]),
                'label': int(labels[i])
            }
            for i in hardest_genuine_idx
        ],
        'spoof': [
            {
                'file': file_names[i],
                'score': float(predictions[i]),
                'confidence': float(confidence[i]),
                'label': int(labels[i])
            }
            for i in hardest_spoof_idx
        ]
    }
    
    return hardest


def generate_comprehensive_report(
    results_dict: Dict,
    output_dir: str = 'outputs/reports'
):
    """
    Generate comprehensive evaluation report with all statistics.
    
    Args:
        results_dict: Dictionary containing all results from multiple runs
        output_dir: Directory to save reports
    
    Example:
        >>> results = run_multiple_seeds_experiment(...)
        >>> generate_comprehensive_report(results)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create report text file
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Main metrics
        f.write("MAIN METRICS (Mean ± Std across runs):\n")
        f.write("-" * 80 + "\n")
        
        for metric in ['eer', 'precision', 'recall', 'f1']:
            if f'{metric}_mean' in results_dict:
                mean = results_dict[f'{metric}_mean']
                std = results_dict[f'{metric}_std']
                f.write(f"{metric.upper():15s}: {mean:.4f} ± {std:.4f}\n")
        
        # Confidence intervals (if available)
        if 'confidence_intervals' in results_dict:
            f.write("\n\nCONFIDENCE INTERVALS (95%):\n")
            f.write("-" * 80 + "\n")
            for metric, ci in results_dict['confidence_intervals'].items():
                f.write(f"{metric:15s}: [{ci['lower']:.4f}, {ci['upper']:.4f}]\n")
        
        # Statistical tests
        if 'statistical_tests' in results_dict:
            f.write("\n\nSTATISTICAL TESTS:\n")
            f.write("-" * 80 + "\n")
            for test_name, test_result in results_dict['statistical_tests'].items():
                f.write(f"\n{test_name}:\n")
                f.write(f"  Statistic: {test_result.get('statistic', 'N/A')}\n")
                f.write(f"  P-value: {test_result.get('p_value', 'N/A')}\n")
                f.write(f"  Result: {test_result.get('interpretation', 'N/A')}\n")
        
        # Hardest samples
        if 'hardest_samples' in results_dict:
            f.write("\n\nHARDEST SAMPLES (Lowest Confidence Correct Predictions):\n")
            f.write("-" * 80 + "\n")
            
            hardest = results_dict['hardest_samples']
            
            f.write("\nGenuine Samples:\n")
            for i, sample in enumerate(hardest['genuine'][:10], 1):
                f.write(f"  {i}. {sample['file']}: score={sample['score']:.4f}, "
                       f"confidence={sample['confidence']:.4f}\n")
            
            f.write("\nSpoof Samples:\n")
            for i, sample in enumerate(hardest['spoof'][:10], 1):
                f.write(f"  {i}. {sample['file']}: score={sample['score']:.4f}, "
                       f"confidence={sample['confidence']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✓ Comprehensive report saved to: {report_path}")


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: str = 'outputs/confusion_matrix.png',
    threshold: float = 0.5
):
    """
    Plot confusion matrix.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        save_path: Path to save figure
        threshold: Classification threshold
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    pred_binary = (predictions.flatten() >= threshold).astype(int)
    labels_binary = labels.flatten().astype(int)
    
    cm = confusion_matrix(labels_binary, pred_binary)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Genuine', 'Spoof'],
                yticklabels=['Genuine', 'Spoof'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {save_path}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Example: Enhanced Reporting")
    
    # Simulate data
    np.random.seed(42)
    predictions = np.random.rand(1000)
    labels = np.random.randint(0, 2, 1000)
    file_names = [f"sample_{i}.wav" for i in range(1000)]
    
    # Plot distribution
    plot_prediction_distribution(predictions, labels)
    
    # Find hardest samples
    hardest = find_hardest_samples(predictions, labels, file_names, n_samples=10)
    print("\nTop 3 Hardest Genuine Samples:")
    for sample in hardest['genuine'][:3]:
        print(f"  {sample['file']}: score={sample['score']:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(predictions, labels)



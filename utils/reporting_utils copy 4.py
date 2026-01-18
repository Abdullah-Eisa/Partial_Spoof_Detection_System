
"""
Enhanced Reporting Utilities
Location: utils/reporting_utils.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Tuple
import os


def _sigmoid(x):
    """Apply sigmoid to convert logits to probabilities."""
    return 1 / (1 + np.exp(-x))


def plot_prediction_distribution(
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: str = 'outputs/prediction_distribution.png',
    bins: int = 50,
    use_logits: bool = True
):
    """
    Plot distribution of prediction scores for genuine vs spoof samples.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        labels: Ground truth labels (0=genuine, 1=spoof)
        save_path: Path to save figure
        bins: Number of histogram bins
        use_logits: If True, predictions are logits and will be converted to probs
    
    Example:
        >>> plot_prediction_distribution(predictions, labels, use_logits=True)
    """
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Convert logits to probabilities if needed
    if use_logits:
        predictions_prob = _sigmoid(predictions)
        x_label = 'Prediction Probability'
    else:
        predictions_prob = predictions
        x_label = 'Prediction Score'
    
    # Separate by class
    genuine_preds = predictions_prob[labels == 0]
    spoof_preds = predictions_prob[labels == 1]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot histogram
    axes[0].hist(genuine_preds, bins=bins, alpha=0.6, label='Genuine', color='green', density=True)
    axes[0].hist(spoof_preds, bins=bins, alpha=0.6, label='Spoof', color='red', density=True)
    axes[0].axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold=0.5')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution of Prediction Scores')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])  # Probabilities are in [0,1]
    
    # Plot box plot
    axes[1].boxplot([genuine_preds, spoof_preds], labels=['Genuine', 'Spoof'], 
                    vert=False, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.6))
    axes[1].axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, label='Threshold=0.5')
    axes[1].set_xlabel(x_label)
    axes[1].set_title('Box Plot of Prediction Scores by Class')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Prediction distribution saved to: {save_path}")
    
    # Print statistics
    print("\nPrediction Statistics (Probabilities):")
    print(f"Genuine - Mean: {genuine_preds.mean():.4f}, Std: {genuine_preds.std():.4f}")
    print(f"Spoof   - Mean: {spoof_preds.mean():.4f}, Std: {spoof_preds.std():.4f}")
    
    # Print logit statistics too (for debugging)
    if use_logits:
        genuine_logits = predictions[labels == 0]
        spoof_logits = predictions[labels == 1]
        print("\nPrediction Statistics (Raw Logits):")
        print(f"Genuine - Mean: {genuine_logits.mean():.4f}, Std: {genuine_logits.std():.4f}")
        print(f"Spoof   - Mean: {spoof_logits.mean():.4f}, Std: {spoof_logits.std():.4f}")


def find_hardest_samples(
    predictions: np.ndarray,
    labels: np.ndarray,
    file_names: List[str],
    n_samples: int = 20,
    threshold: float = 0.5,
    use_logits: bool = True
) -> Dict:
    """
    Identify hardest samples (lowest confidence correct predictions).
    
    Args:
        predictions: Model predictions (logits or probabilities)
        labels: Ground truth labels
        file_names: List of file names
        n_samples: Number of hardest samples to return
        threshold: Classification threshold (in probability space)
        use_logits: If True, predictions are logits and will be converted
    
    Returns:
        Dictionary with hardest genuine and spoof samples
    
    Example:
        >>> hardest = find_hardest_samples(preds, labels, file_names, use_logits=True)
        >>> print("Hardest genuine samples:")
        >>> for sample in hardest['genuine'][:5]:
        ...     print(f"  {sample['file']}: prob={sample['prob']:.4f}, confidence={sample['confidence']:.4f}")
    """
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Store original logits for output
    predictions_logits = predictions.copy() if use_logits else None
    
    # Convert logits to probabilities if needed
    if use_logits:
        predictions_prob = _sigmoid(predictions)
    else:
        predictions_prob = predictions
    
    # Find correct predictions
    pred_binary = (predictions_prob >= threshold).astype(int)
    correct_mask = (pred_binary == labels)
    
    # Calculate confidence using LOGITS to avoid underflow issues
    # For logits: threshold=0.5 in prob space = logit=0.0
    # Confidence = |logit - 0| = |logit|
    # Smaller |logit| = closer to decision boundary = harder
    if use_logits:
        confidence = np.abs(predictions)  # Distance from logit=0 (decision boundary)
    else:
        confidence = np.abs(predictions_prob - threshold)  # Distance from prob=0.5
    
    # Hardest genuine samples (correct, but close to threshold)
    genuine_mask = (labels == 0) & correct_mask
    genuine_indices = np.where(genuine_mask)[0]
    
    if len(genuine_indices) > 0:
        genuine_confidence = confidence[genuine_indices]
        # Sort by ASCENDING confidence → lowest first (closest to threshold)
        hardest_genuine_idx = genuine_indices[np.argsort(genuine_confidence)[:n_samples]]
    else:
        hardest_genuine_idx = []
    
    # Hardest spoof samples (correct, but close to threshold)
    spoof_mask = (labels == 1) & correct_mask
    spoof_indices = np.where(spoof_mask)[0]
    
    if len(spoof_indices) > 0:
        spoof_confidence = confidence[spoof_indices]
        # Sort by ASCENDING confidence → lowest first (closest to threshold)
        hardest_spoof_idx = spoof_indices[np.argsort(spoof_confidence)[:n_samples]]
    else:
        hardest_spoof_idx = []
    
    hardest = {
        'genuine': [
            {
                'file': file_names[i],
                'logit': float(predictions[i]) if use_logits else None,
                'prob': float(predictions_prob[i]),
                'confidence': float(confidence[i]),
                'label': int(labels[i])
            }
            for i in hardest_genuine_idx
        ],
        'spoof': [
            {
                'file': file_names[i],
                'logit': float(predictions[i]) if use_logits else None,
                'prob': float(predictions_prob[i]),
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
            
            f.write("\nGenuine Samples (closest to threshold 0.5):\n")
            for i, sample in enumerate(hardest['genuine'][:100], 1):
                logit_str = f", logit={sample['logit']:.4f}" if sample.get('logit') is not None else ""
                f.write(f"  {i}. {sample['file']}: prob={sample['prob']:.4f}{logit_str}, "
                       f"confidence={sample['confidence']:.4f}\n")
            
            f.write("\nSpoof Samples (closest to threshold 0.5):\n")
            for i, sample in enumerate(hardest['spoof'][:100], 1):
                logit_str = f", logit={sample['logit']:.4f}" if sample.get('logit') is not None else ""
                f.write(f"  {i}. {sample['file']}: prob={sample['prob']:.4f}{logit_str}, "
                       f"confidence={sample['confidence']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✓ Comprehensive report saved to: {report_path}")


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: str = 'outputs/confusion_matrix.png',
    threshold: float = 0.5,
    use_logits: bool = True
):
    """
    Plot confusion matrix.
    
    Args:
        predictions: Model predictions
        labels: Ground truth labels
        save_path: Path to save figure
        threshold: Classification threshold
        use_logits: If True, convert logits to probabilities first
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Convert logits to probabilities if needed
    if use_logits:
        predictions = _sigmoid(predictions)
    
    pred_binary = (predictions >= threshold).astype(int)
    labels_binary = labels.astype(int)
    
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
# Example Usage in __main__
# ============================================================================

if __name__ == "__main__":
    """
    Standalone script to generate reporting visualizations.
    
    Usage:
        python -m utils.reporting_utils --config config/default_config.yaml
    """
    import argparse
    from utils.config_manager import ConfigManager
    from model import initialize_models
    from preprocess import initialize_data_loader
    import torch
    
    parser = argparse.ArgumentParser(description='Enhanced Reporting on Inference Results')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs/reports4',
                       help='Output directory for reports')
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigManager(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("GENERATING ENHANCED REPORTS")
    print("="*80)
    
    # Load model and feature extractor
    print("\n1. Loading model...")
    model, feature_extractor, _ = initialize_models(
        ssl_ckpt_path=config['paths']['ssl_checkpoint'],
        save_feature_extractor=False,
        feature_dim=config['model']['feature_dim'],
        num_heads=config['model']['num_heads'],
        hidden_dim=config['model']['hidden_dim'],
        max_dropout=config['model']['max_dropout'],
        depthwise_conv_kernel_size=config['model']['depthwise_conv_kernel_size'],
        conformer_layers=config['model']['conformer_layers'],
        max_pooling_factor=config['model']['max_pooling_factor'],
        LEARNING_RATE=0.0001,
        DEVICE=device
    )

    checkpoint = torch.load(config['paths']['ps_model_checkpoint'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get test data loader
    test_loader = initialize_data_loader(
        dataset_name=config['data']['dataset_name'],
        data_path=config['data']['eval_data_path'],
        labels_path=config['data']['eval_labels_path'],
        BATCH_SIZE=config['inference']['batch_size'],
        shuffle=False,
        num_workers=config['inference'].get('num_workers', 4),
        prefetch_factor=config['inference'].get('prefetch_factor', 2),
        pin_memory=config['inference'].get('pin_memory', True)
    )
    
    # Collect predictions and labels
    print("2. Collecting predictions, labels, and file names...")
    all_predictions = []
    all_labels = []
    all_files = []
    
    with torch.no_grad():
        for batch in test_loader:
            waveforms = batch['waveform'].to(device)
            labels = batch['label']
            
            features_output = feature_extractor(waveforms)
            if isinstance(features_output, dict):
                features = features_output['hidden_states'][-1]
            else:
                features = features_output
            
            lengths = torch.full((features.size(0),), features.size(1), 
                               dtype=torch.int16, device=device)
            outputs = model(features, lengths, dropout_prob=0.0)
            
            all_predictions.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_files.extend(batch['file_name'])
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # Generate reports
    print("\n3. Generating visualizations...")
    
    # Plot prediction distribution (WITH LOGIT CONVERSION)
    print("   - Prediction distribution...")
    plot_prediction_distribution(
        predictions, labels,
        save_path=os.path.join(args.output_dir, 'prediction_distribution.png'),
        use_logits=True  # IMPORTANT: Tell it these are logits
    )
    
    # Find hardest samples (WITH LOGIT CONVERSION)
    print("   - Finding hardest samples...")
    hardest = find_hardest_samples(
        predictions, labels, all_files, 
        n_samples=100,
        use_logits=True  # IMPORTANT: Tell it these are logits
    )
    
    print("\n   Top 5 Hardest Genuine Samples:")
    for i, sample in enumerate(hardest['genuine'][:5], 1):
        print(f"     {i}. {sample['file']}: prob={sample['prob']:.4f}, "
              f"logit={sample['logit']:.4f}, confidence={sample['confidence']:.4f}")
    
    print("\n   Top 5 Hardest Spoof Samples:")
    for i, sample in enumerate(hardest['spoof'][:5], 1):
        print(f"     {i}. {sample['file']}: prob={sample['prob']:.4f}, "
              f"logit={sample['logit']:.4f}, confidence={sample['confidence']:.4f}")
    
    # Plot confusion matrix (WITH LOGIT CONVERSION)
    print("\n   - Confusion matrix...")
    plot_confusion_matrix(
        predictions, labels,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png'),
        use_logits=True  # IMPORTANT: Tell it these are logits
    )
    
    # Save hardest samples to JSON
    import json
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    hardest_path = os.path.join(args.output_dir, f'hardest_samples_{timestamp}.json')
    with open(hardest_path, 'w') as f:
        json.dump(hardest, f, indent=2)
    print(f"\n✓ Hardest samples saved to: {hardest_path}")
    
    print(f"\n✓ All reports saved to: {args.output_dir}")

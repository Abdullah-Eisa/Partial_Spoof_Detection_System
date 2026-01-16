
"""
Comprehensive Evaluation Script
Integrates all analysis requirements:
1. Statistical Testing (Bootstrap CI, McNemar's, Bonferroni)
2. Enhanced Reporting (Distributions, Hardest Samples)
3. Attention Visualization
4. Gradient-based Analysis (Integrated Gradients)
5. Cluster Analysis

Usage:
    python run_comprehensive_evaluation.py --config config/default_config.yaml
"""

import os
import argparse
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List

# Import utilities
from utils.config_manager import ConfigManager
from utils.statistical_testing import (
    bootstrap_confidence_interval,
    mcnemar_test,
    bonferroni_correction,
    run_multiple_seeds_experiment,
    save_statistical_results
)
from utils.reporting_utils import (
    plot_prediction_distribution,
    find_hardest_samples,
    generate_comprehensive_report,
    plot_confusion_matrix
)
from utils.attention_visualization import (
    AttentionExtractor,
    visualize_attention_on_spectrogram,
    compare_attention_patterns
)
from utils.gradient_analysis import (
    IntegratedGradients,
    analyze_boundary_focus,
    compare_genuine_vs_pf_focus
)
from utils.cluster_analysis import (
    EmbeddingExtractor,
    visualize_embeddings_tsne,
    compute_cluster_metrics,
    analyze_misclassified_embeddings
)
from utils.utils import compute_eer

# Import model components
from model import BinarySpoofingClassificationModel
from feature_extractors import FeatureExtractorFactory
from preprocess import initialize_data_loader
from inference import inference


def run_statistical_analysis(
    predictions: np.ndarray,
    labels: np.ndarray,
    output_dir: str = 'outputs/statistical_analysis'
):
    """Run all statistical tests."""
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Bootstrap Confidence Intervals
    print("\n1. Computing Bootstrap Confidence Intervals (1000 samples)...")
    ci_eer = bootstrap_confidence_interval(
        predictions, labels,
        metric_fn=lambda p, l: compute_eer(torch.tensor(p), torch.tensor(l))[0],
        n_bootstrap=1000
    )
    
    print(f"   EER: {ci_eer['mean']:.4f} [{ci_eer['lower']:.4f}, {ci_eer['upper']:.4f}]")
    
    # Save results
    results = {
        'confidence_intervals': {
            'eer': ci_eer
        }
    }
    
    save_statistical_results(
        results,
        os.path.join(output_dir, 'bootstrap_results.json')
    )
    
    return results


def run_reporting_analysis(
    predictions: np.ndarray,
    labels: np.ndarray,
    file_names: List[str],
    output_dir: str = 'outputs/reporting'
):
    """Generate comprehensive reports and visualizations."""
    print("\n" + "="*80)
    print("REPORTING & VISUALIZATION")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Prediction Distribution
    print("\n1. Plotting prediction distributions...")
    plot_prediction_distribution(
        predictions, labels,
        save_path=os.path.join(output_dir, 'prediction_distribution.png')
    )
    
    # 2. Confusion Matrix
    print("\n2. Generating confusion matrix...")
    plot_confusion_matrix(
        predictions, labels,
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # 3. Hardest Samples
    print("\n3. Finding hardest samples...")
    hardest = find_hardest_samples(
        predictions, labels, file_names, n_samples=20
    )
    
    print(f"   Top 5 Hardest Genuine Samples:")
    for i, sample in enumerate(hardest['genuine'][:5], 1):
        print(f"     {i}. {sample['file']}: score={sample['score']:.4f}")
    
    print(f"\n   Top 5 Hardest Spoof Samples:")
    for i, sample in enumerate(hardest['spoof'][:5], 1):
        print(f"     {i}. {sample['file']}: score={sample['score']:.4f}")
    
    return {'hardest_samples': hardest}


def run_attention_analysis(
    model,
    feature_extractor,
    sample_audio_path: str,
    segment_boundaries: List[tuple] = None,
    output_dir: str = 'outputs/attention_analysis',
    device: str = 'cpu'
):
    """Analyze attention patterns."""
    print("\n" + "="*80)
    print("ATTENTION ANALYSIS")
    print("="*80)
    
    if not os.path.exists(sample_audio_path):
        print(f"⚠️  Sample audio not found: {sample_audio_path}")
        print("   Skipping attention analysis.")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    import torchaudio
    
    # Load audio
    waveform, sr = torchaudio.load(sample_audio_path)
    waveform = waveform.to(device)
    
    # Extract attention
    print("\n1. Extracting attention weights...")
    extractor = AttentionExtractor(model)
    attention_dict = extractor.extract_attention(waveform, feature_extractor)
    
    if 'time_pooling' in attention_dict:
        attention = attention_dict['time_pooling'].squeeze().numpy()
        attention = (attention - attention.min()) / (attention.max() - attention.min())
        
        print("2. Visualizing attention on spectrogram...")
        visualize_attention_on_spectrogram(
            waveform.squeeze().cpu().numpy(),
            attention,
            sample_rate=sr,
            segment_boundaries=segment_boundaries,
            save_path=os.path.join(output_dir, 'attention_spectrogram.png')
        )
    
    extractor.remove_hooks()
    print("✓ Attention analysis complete")


def run_gradient_analysis(
    model,
    feature_extractor,
    sample_audio_path: str,
    segment_boundaries: List[tuple] = None,
    output_dir: str = 'outputs/gradient_analysis',
    device: str = 'cpu'
):
    """Run integrated gradients analysis."""
    print("\n" + "="*80)
    print("GRADIENT-BASED ANALYSIS (INTEGRATED GRADIENTS)")
    print("="*80)
    
    if not os.path.exists(sample_audio_path):
        print(f"⚠️  Sample audio not found: {sample_audio_path}")
        print("   Skipping gradient analysis.")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    import torchaudio
    
    # Load audio
    waveform, sr = torchaudio.load(sample_audio_path)
    waveform = waveform.to(device)
    
    # Compute integrated gradients
    print("\n1. Computing integrated gradients (50 steps)...")
    ig = IntegratedGradients(model, feature_extractor)
    attributions = ig.compute_integrated_gradients(waveform, n_steps=50)
    
    print("2. Visualizing attributions...")
    ig.visualize_attributions(
        waveform.squeeze().cpu().numpy(),
        attributions.squeeze().cpu().numpy(),
        sample_rate=sr,
        segment_boundaries=segment_boundaries,
        save_path=os.path.join(output_dir, 'integrated_gradients.png')
    )
    
    # Analyze boundary focus if boundaries provided
    if segment_boundaries:
        print("\n3. Analyzing focus on segment boundaries...")
        # Convert time boundaries to sample indices
        frame_boundaries = [
            (int(start * sr), int(end * sr))
            for start, end in segment_boundaries
        ]
        
        stats = analyze_boundary_focus(
            attributions.squeeze().cpu().numpy(),
            frame_boundaries,
            window_size=int(0.1 * sr)  # 100ms window
        )
        
        print(f"   Boundary mean: {stats['boundary_mean']:.4f}")
        print(f"   Non-boundary mean: {stats['non_boundary_mean']:.4f}")
        print(f"   Focus ratio: {stats['focus_ratio']:.2f}")
    
    print("✓ Gradient analysis complete")


def run_cluster_analysis(
    model,
    feature_extractor,
    dataloader,
    predictions: np.ndarray,
    labels: np.ndarray,
    file_names: List[str],
    output_dir: str = 'outputs/cluster_analysis',
    device: str = 'cpu'
):
    """Run clustering and embedding analysis."""
    print("\n" + "="*80)
    print("CLUSTER ANALYSIS")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Extract embeddings
    print("\n1. Extracting embeddings from model...")
    extractor = EmbeddingExtractor(model, layer_name='pooling')
    embeddings, emb_labels, emb_files = extractor.extract_embeddings(
        dataloader, feature_extractor, device
    )
    
    print(f"   Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # 2. Visualize with t-SNE
    print("\n2. Computing t-SNE visualization...")
    visualize_embeddings_tsne(
        embeddings, emb_labels,
        save_path=os.path.join(output_dir, 'tsne_visualization.png')
    )
    
    # 3. Compute cluster metrics
    print("\n3. Computing cluster quality metrics...")
    metrics = compute_cluster_metrics(embeddings, emb_labels)
    
    print(f"   Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"   Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
    print(f"   Separability Ratio: {metrics['separability_ratio']:.4f}")
    
    # 4. Analyze misclassified samples
    print("\n4. Analyzing misclassified samples in embedding space...")
    pred_binary = (predictions >= 0.5).astype(int)
    analyze_misclassified_embeddings(
        embeddings[:len(predictions)],  # Match size
        labels.astype(int),
        pred_binary,
        file_names,
        save_path=os.path.join(output_dir, 'misclassified_analysis.png')
    )
    
    extractor.remove_hook()
    print("✓ Cluster analysis complete")
    
    return metrics


def main(args):
    """Main evaluation pipeline."""
    print("\n" + "="*100)
    print("COMPREHENSIVE EVALUATION PIPELINE")
    print("="*100)
    
    # Load configuration
    config = ConfigManager(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = f'outputs/comprehensive_evaluation_{timestamp}'
    os.makedirs(output_root, exist_ok=True)
    
    print(f"\nOutput directory: {output_root}")
    
    # Run inference to get predictions
    print("\n" + "="*100)
    print("RUNNING INFERENCE")
    print("="*100)
    
    inference_results = inference(config, show_model_info=False)
    
    # Load model and feature extractor for analysis
    from model import initialize_models
    model, feature_extractor, _ = initialize_models(
        config, save_feature_extractor=False, LEARNING_RATE=0.0001, DEVICE=device
    )
    
    # Load checkpoint
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
    
    # Collect all predictions and labels
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
    
    # Initialize results dictionary
    comprehensive_results = {
        'timestamp': timestamp,
        'config': dict(config),
        'inference_metrics': inference_results
    }
    
    # 1. Statistical Analysis
    if not args.skip_statistical:
        stat_results = run_statistical_analysis(
            predictions, labels,
            output_dir=os.path.join(output_root, 'statistical_analysis')
        )
        comprehensive_results.update(stat_results)
    
    # 2. Reporting & Visualization
    if not args.skip_reporting:
        report_results = run_reporting_analysis(
            predictions, labels, all_files,
            output_dir=os.path.join(output_root, 'reporting')
        )
        comprehensive_results.update(report_results)
    
    # 3. Attention Analysis (if sample provided)
    if not args.skip_attention and args.sample_audio:
        run_attention_analysis(
            model, feature_extractor,
            args.sample_audio,
            segment_boundaries=args.segment_boundaries,
            output_dir=os.path.join(output_root, 'attention_analysis'),
            device=device
        )
    
    # 4. Gradient Analysis (if sample provided)
    if not args.skip_gradient and args.sample_audio:
        run_gradient_analysis(
            model, feature_extractor,
            args.sample_audio,
            segment_boundaries=args.segment_boundaries,
            output_dir=os.path.join(output_root, 'gradient_analysis'),
            device=device
        )
    
    # 5. Cluster Analysis
    if not args.skip_cluster:
        cluster_metrics = run_cluster_analysis(
            model, feature_extractor, test_loader,
            predictions, labels, all_files,
            output_dir=os.path.join(output_root, 'cluster_analysis'),
            device=device
        )
        comprehensive_results['cluster_metrics'] = cluster_metrics
    
    # Generate final comprehensive report
    print("\n" + "="*100)
    print("GENERATING FINAL REPORT")
    print("="*100)
    
    generate_comprehensive_report(
        comprehensive_results,
        output_dir=output_root
    )
    
    print("\n" + "="*100)
    print("✓ COMPREHENSIVE EVALUATION COMPLETE")
    print("="*100)
    print(f"\nAll results saved to: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive Model Evaluation')
    
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--sample-audio', type=str, default=None,
                      help='Path to sample audio for attention/gradient analysis')
    parser.add_argument('--segment-boundaries', type=float, nargs='+', default=None,
                      help='Segment boundaries in seconds (e.g., 0.5 1.2 2.0 2.8)')
    
    # Skip options
    parser.add_argument('--skip-statistical', action='store_true',
                      help='Skip statistical testing')
    parser.add_argument('--skip-reporting', action='store_true',
                      help='Skip reporting and visualization')
    parser.add_argument('--skip-attention', action='store_true',
                      help='Skip attention analysis')
    parser.add_argument('--skip-gradient', action='store_true',
                      help='Skip gradient analysis')
    parser.add_argument('--skip-cluster', action='store_true',
                      help='Skip cluster analysis')
    
    args = parser.parse_args()
    
    # Parse segment boundaries if provided
    if args.segment_boundaries and len(args.segment_boundaries) % 2 == 0:
        args.segment_boundaries = [
            (args.segment_boundaries[i], args.segment_boundaries[i+1])
            for i in range(0, len(args.segment_boundaries), 2)
        ]
    
    main(args)

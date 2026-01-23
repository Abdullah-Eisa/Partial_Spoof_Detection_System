
"""
Cluster Analysis for Learned Representations
Location: utils/cluster_analysis.py

Analyzes:
1. How embeddings cluster by class (genuine/spoof/PF)
2. Within-class and between-class distances
3. Visualization with t-SNE/UMAP
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import os


class EmbeddingExtractor:
    """Extract embeddings from model's intermediate layers."""
    
    def __init__(self, model: nn.Module, layer_name: str = 'pooling'):
        """
        Args:
            model: BinarySpoofingClassificationModel
            layer_name: Name of layer to extract embeddings from
        """
        self.model = model
        self.layer_name = layer_name
        self.embeddings = {}
        self.hook = None
        self._register_hook()
    
    def _register_hook(self):
        """Register forward hook to capture embeddings."""
        
        def hook(module, input, output):
            self.embeddings['embedding'] = output.detach().cpu()
    
        # Get the layer based on layer_name
        if self.layer_name == 'pooling':
            self.hook = self.model.pooling.register_forward_hook(hook)
            print(f"✓ Extracting embeddings from: model.pooling")
            print(f"  Layer type: {type(self.model.pooling).__name__}")
            print(f"  Output shape: (batch_size, hidden_dim)")
            
        elif self.layer_name == 'conformer':
            self.hook = self.model.conformer.register_forward_hook(hook)
            print(f"✓ Extracting embeddings from: model.conformer")
            print(f"  Layer type: {type(self.model.conformer).__name__}")
            print(f"  Output shape: (batch_size, time_dim, hidden_dim)")
            
        elif self.layer_name == 'max_pooling':
            self.hook = self.model.max_pooling.register_forward_hook(hook)
            print(f"✓ Extracting embeddings from: model.max_pooling")
            print(f"  Layer type: {type(self.model.max_pooling).__name__}")
            print(f"  Output shape: (batch_size, feature_dim, time_dim)")
            
        elif hasattr(self.model, self.layer_name):
            layer = getattr(self.model, self.layer_name)
            self.hook = layer.register_forward_hook(hook)
            print(f"✓ Extracting embeddings from: model.{self.layer_name}")
            print(f"  Layer type: {type(layer).__name__}")
        else:
            raise ValueError(f"Layer '{self.layer_name}' not found in model. "
                           f"Available options: 'pooling', 'conformer', 'max_pooling'")


    # def extract_embeddings(
    #     self,
    #     dataloader,
    #     feature_extractor: nn.Module,
    #     device: str = 'cpu'
    # ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    #     """
    #     Extract embeddings for entire dataset.
        
    #     Args:
    #         dataloader: DataLoader
    #         feature_extractor: Feature extraction model
    #         device: Device to run on
        
    #     Returns:
    #         (embeddings, labels, file_names)
    #     """
    #     all_embeddings = []
    #     all_labels = []
    #     all_files = []
        
    #     self.model.eval()
    #     feature_extractor.eval()
        
    #     with torch.no_grad():
    #         for batch in dataloader:
    #             waveforms = batch['waveform'].to(device)
    #             labels = batch['label'].to(device)
                
    #             # Extract features
    #             features_output = feature_extractor(waveforms)
    #             if isinstance(features_output, dict):
    #                 features = features_output['hidden_states'][-1]
    #             else:
    #                 features = features_output
                
    #             lengths = torch.full((features.size(0),), features.size(1), 
    #                                dtype=torch.int16, device=device)
                
    #             # Forward pass (embeddings captured by hook)
    #             _ = self.model(features, lengths, dropout_prob=0.0)
                
    #             # Store results
    #             all_embeddings.append(self.embeddings['embedding'].numpy())
    #             all_labels.append(labels.cpu().numpy())
    #             all_files.extend(batch['file_name'])
        
    #     embeddings = np.vstack(all_embeddings)
    #     labels = np.concatenate(all_labels)
        
    #     print("✨✨✨✨✨",len(all_files))
    #     print(f"  Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]} , embeddings.shape= {embeddings.shape}, labels.shape= {labels.shape}")

    #     return embeddings, labels, all_files


    def extract_embeddings(
        self,
        dataloader,
        feature_extractor: nn.Module,
        device: str = 'cpu'
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract embeddings for entire dataset.
        
        Args:
            dataloader: DataLoader
            feature_extractor: Feature extraction model
            device: Device to run on
        
        Returns:
            (embeddings, labels, file_names)
        """
        all_embeddings = []
        all_labels = []
        all_files = []
        
        self.model.eval()
        feature_extractor.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                waveforms = batch['waveform'].to(device)
                labels = batch['label'].to(device)
                
                # Extract features
                features_output = feature_extractor(waveforms)
                if isinstance(features_output, dict):
                    features = features_output['hidden_states'][-1]
                else:
                    features = features_output
                
                lengths = torch.full((features.size(0),), features.size(1), 
                                   dtype=torch.int16, device=device)
                
                # Forward pass (embeddings captured by hook)
                _ = self.model(features, lengths, dropout_prob=0.0)
                
                # Store results
                embedding = self.embeddings['embedding'].numpy()
                
                # Handle different shapes based on layer
                if self.layer_name == 'conformer':
                    # Shape: (batch_size, time_dim, hidden_dim) - keep as is
                    all_embeddings.append(embedding)
                elif self.layer_name in ['pooling', 'max_pooling']:
                    # Flatten if needed: (batch_size, hidden_dim) or (batch_size, feature_dim, time_dim)
                    if embedding.ndim == 3:
                        # Flatten (batch_size, dim, time) to (batch_size, dim*time)
                        batch_size = embedding.shape[0]
                        all_embeddings.append(embedding.reshape(batch_size, -1))
                    else:
                        all_embeddings.append(embedding)
                else:
                    all_embeddings.append(embedding)
                
                all_labels.append(labels.cpu().numpy())
                all_files.extend(batch['file_name'])
        
        embeddings = np.vstack(all_embeddings)
        labels = np.concatenate(all_labels)
        
        print(f"✓ Extracted {len(embeddings)} embeddings")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Labels shape: {labels.shape}")

        return embeddings, labels, all_files



    def extract_embeddings_from_feature_extractor(
        self,
        dataloader,
        feature_extractor: nn.Module,
        device: str = 'cpu',
        layer_index: int = -1  # -1 for last hidden state
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract embeddings directly from feature extractor.
        
        Args:
            dataloader: DataLoader
            feature_extractor: Feature extraction model
            device: Device to run on
            layer_index: Which hidden state to extract (-1 for last)
        
        Returns:
            (embeddings, labels, file_names)
        """
        all_embeddings = []
        all_labels = []
        all_files = []
        
        feature_extractor.eval()
        
        print(f"✓ Extracting embeddings from: feature_extractor.hidden_states[{layer_index}]")
        
        with torch.no_grad():
            for batch in dataloader:
                waveforms = batch['waveform'].to(device)
                labels = batch['label'].to(device)
                
                # Extract features
                features_output = feature_extractor(waveforms)
                if isinstance(features_output, dict):
                    features = features_output['hidden_states'][layer_index]
                else:
                    features = features_output
                
                # Pool over time dimension to get fixed-size embedding
                # Mean pooling across time
                embedding = features.mean(dim=1)  # (batch, time, dim) -> (batch, dim)
                
                # Store results
                all_embeddings.append(embedding.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_files.extend(batch['file_name'])
        
        embeddings = np.vstack(all_embeddings)
        labels = np.concatenate(all_labels)
        
        print("✨✨✨✨✨",len(all_files))
        print(f"  Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]} , embeddings.shape= {embeddings.shape}, labels.shape= {labels.shape}")

        return embeddings, labels, all_files



    def remove_hook(self):
        """Remove the registered hook."""
        if self.hook:
            self.hook.remove()

# =============================================================================================================

def visualize_embeddings_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str = 'outputs/tsne_visualization.png',
    perplexity: int = 30,
    n_iter: int = 1000,
    random_state: int = 42
):
    """
    Visualize embeddings using t-SNE.
    
    Args:
        embeddings: Embedding vectors (n_samples, embedding_dim)
        labels: Class labels (n_samples,)
        save_path: Path to save figure
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        random_state: Random seed
    """
    print("Computing t-SNE projection...")
    
    # Reduce dimensionality with PCA first (faster)
    if embeddings.shape[1] > 50:
        pca = PCA(n_components=50, random_state=random_state)
        embeddings_reduced = pca.fit_transform(embeddings)
        print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        embeddings_reduced = embeddings
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                random_state=random_state, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings_reduced)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Separate by class
    genuine_mask = (labels == 0)
    spoof_mask = (labels == 1)
    
    ax.scatter(embeddings_2d[genuine_mask, 0], embeddings_2d[genuine_mask, 1],
              c='green', label='Genuine', alpha=0.6, s=30)
    ax.scatter(embeddings_2d[spoof_mask, 0], embeddings_2d[spoof_mask, 1],
              c='red', label='Spoof', alpha=0.6, s=30)
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('t-SNE Visualization of Learned Embeddings')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ t-SNE visualization saved to: {save_path}")


def compute_cluster_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute clustering quality metrics.
    
    Args:
        embeddings: Embedding vectors
        labels: Ground truth labels
    
    Returns:
        Dictionary with clustering metrics
    """
    # Silhouette score (higher is better, range [-1, 1])
    silhouette = silhouette_score(embeddings, labels)
    
    # Davies-Bouldin index (lower is better)
    davies_bouldin = davies_bouldin_score(embeddings, labels)
    
    # Within-class and between-class distances
    genuine_embeddings = embeddings[labels == 0]
    spoof_embeddings = embeddings[labels == 1]
    
    # Within-class variance
    within_genuine = np.var(genuine_embeddings, axis=0).mean()
    within_spoof = np.var(spoof_embeddings, axis=0).mean()
    
    # Between-class distance (centroid distance)
    centroid_genuine = genuine_embeddings.mean(axis=0)
    centroid_spoof = spoof_embeddings.mean(axis=0)
    between_distance = np.linalg.norm(centroid_genuine - centroid_spoof)
    
    return {
        'silhouette_score': float(silhouette),
        'davies_bouldin_index': float(davies_bouldin),
        'within_class_variance_genuine': float(within_genuine),
        'within_class_variance_spoof': float(within_spoof),
        'between_class_distance': float(between_distance),
        'separability_ratio': float(between_distance / (within_genuine + within_spoof + 1e-8))
    }


def analyze_misclassified_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    file_names: List[str],
    save_path: str = 'outputs/misclassified_analysis.png'
):
    """
    Analyze embeddings of misclassified samples.
    
    Args:
        embeddings: Embedding vectors
        labels: Ground truth labels
        predictions: Model predictions (binary)
        file_names: List of file names
        save_path: Path to save figure
    """
    # Identify misclassified samples
    correct_mask = (predictions == labels)
    misclassified_mask = ~correct_mask
    
    # Reduce dimensionality
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Correctly classified (background)
    ax.scatter(embeddings_2d[correct_mask, 0], embeddings_2d[correct_mask, 1],
              c='lightgray', label='Correct', alpha=0.3, s=20)
    
    # Misclassified genuine (false positive)
    fp_mask = (labels == 0) & misclassified_mask
    ax.scatter(embeddings_2d[fp_mask, 0], embeddings_2d[fp_mask, 1],
              c='orange', label='False Positive', alpha=0.8, s=50, marker='^')
    
    # Misclassified spoof (false negative)
    fn_mask = (labels == 1) & misclassified_mask
    ax.scatter(embeddings_2d[fn_mask, 0], embeddings_2d[fn_mask, 1],
              c='purple', label='False Negative', alpha=0.8, s=50, marker='v')
    
    ax.set_xlabel('PCA Dimension 1')
    ax.set_ylabel('PCA Dimension 2')
    ax.set_title('Misclassified Samples in Embedding Space')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Misclassified analysis saved to: {save_path}")
    
    # Print statistics
    print(f"\nMisclassification Statistics:")
    print(f"  False Positives: {fp_mask.sum()}")
    print(f"  False Negatives: {fn_mask.sum()}")
    print(f"  Total Misclassified: {misclassified_mask.sum()}")


def compare_embedding_distributions(
    embeddings_genuine: np.ndarray,
    embeddings_spoof: np.ndarray,
    embeddings_pf: Optional[np.ndarray] = None,
    save_path: str = 'outputs/embedding_distributions.png'
):
    """
    Compare embedding distributions across classes.
    
    Args:
        embeddings_genuine: Genuine embeddings
        embeddings_spoof: Spoof embeddings
        embeddings_pf: Partial-fake embeddings (optional)
        save_path: Path to save figure
    """
    # Compute norms
    norms_genuine = np.linalg.norm(embeddings_genuine, axis=1)
    norms_spoof = np.linalg.norm(embeddings_spoof, axis=1)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Embedding norm distributions
    axes[0].hist(norms_genuine, bins=50, alpha=0.6, label='Genuine', color='green', density=True)
    axes[0].hist(norms_spoof, bins=50, alpha=0.6, label='Spoof', color='red', density=True)
    
    if embeddings_pf is not None:
        norms_pf = np.linalg.norm(embeddings_pf, axis=1)
        axes[0].hist(norms_pf, bins=50, alpha=0.6, label='Partial-Fake', color='orange', density=True)
    
    axes[0].set_xlabel('Embedding Norm')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Distribution of Embedding Norms')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: First principal component
    all_embeddings = [embeddings_genuine, embeddings_spoof]
    if embeddings_pf is not None:
        all_embeddings.append(embeddings_pf)
    
    combined = np.vstack(all_embeddings)
    pca = PCA(n_components=1)
    pc1_all = pca.fit_transform(combined).flatten()
    
    n_genuine = len(embeddings_genuine)
    n_spoof = len(embeddings_spoof)
    
    pc1_genuine = pc1_all[:n_genuine]
    pc1_spoof = pc1_all[n_genuine:n_genuine + n_spoof]
    
    axes[1].hist(pc1_genuine, bins=50, alpha=0.6, label='Genuine', color='green', density=True)
    axes[1].hist(pc1_spoof, bins=50, alpha=0.6, label='Spoof', color='red', density=True)
    
    if embeddings_pf is not None:
        pc1_pf = pc1_all[n_genuine + n_spoof:]
        axes[1].hist(pc1_pf, bins=50, alpha=0.6, label='Partial-Fake', color='orange', density=True)
    
    axes[1].set_xlabel('First Principal Component')
    axes[1].set_ylabel('Density')
    axes[1].set_title(f'Distribution Along First PC (Variance Explained: {pca.explained_variance_ratio_[0]:.3f})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Embedding distribution comparison saved to: {save_path}")



if __name__ == "__main__":
    """
    Standalone script to run cluster analysis on model embeddings.
    
    Usage:
        python -m utils.cluster_analysis --config config/default_config.yaml
    """
    import argparse
    import torch
    from utils.config_manager import ConfigManager
    from model import initialize_models
    from preprocess import initialize_data_loader
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Cluster Analysis on Model Embeddings')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs/cluster_analysis',
                       help='Output directory for visualizations')
    # parser.add_argument('--layer-name', type=str, default='pooling',
    #                    help='Layer name to extract embeddings from')

    parser.add_argument('--layer-name', type=str, default='pooling',
                       help='Layer name to extract embeddings from: '
                            '"pooling" (after SelfWeightedPooling), '
                            '"conformer" (after Conformer), '
                            '"max_pooling" (after max pooling), '
                            'or "feature_extractor" (from wav2vec2)')

    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--n-iter', type=int, default=1000,
                       help='t-SNE number of iterations')
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigManager(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("CLUSTER ANALYSIS ON MODEL EMBEDDINGS")
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
    feature_extractor.eval()
    
    # Get test data loader
    print("\n2. Loading test data...")
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
    
    # Extract embeddings
    print(f"\n3. Extracting embeddings from layer '{args.layer_name}'...")
    extractor = EmbeddingExtractor(model, layer_name=args.layer_name)

    # embeddings, labels, file_names = extractor.extract_embeddings(
    #     test_loader, feature_extractor, device=device
    # )
    
    # Option 2: Extract from feature extractor
    embeddings, labels, file_names = extractor.extract_embeddings_from_feature_extractor(
        test_loader, feature_extractor, device=device, layer_index=-1
    )

    print(f"   Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Visualize with t-SNE
    print("\n4. Computing t-SNE visualization...")
    visualize_embeddings_tsne(
        embeddings, labels,
        save_path=os.path.join(args.output_dir, f'tsne_visualization_{timestamp}.png'),
        perplexity=args.perplexity,
        n_iter=args.n_iter
    )
    
    # 2. Compute cluster metrics
    print("\n5. Computing cluster quality metrics...")
    metrics = compute_cluster_metrics(embeddings, labels)
    
    print(f"\nCluster Quality Metrics:")
    print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
    print(f"  Within-class variance (Genuine): {metrics['within_class_variance_genuine']:.4f}")
    print(f"  Within-class variance (Spoof): {metrics['within_class_variance_spoof']:.4f}")
    print(f"  Between-class distance: {metrics['between_class_distance']:.4f}")
    print(f"  Separability Ratio: {metrics['separability_ratio']:.4f}")
    
    # Save metrics to JSON
    import json
    metrics_path = os.path.join(args.output_dir, f'cluster_metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # 3. Collect predictions for misclassification analysis
    print("\n6. Collecting model predictions...")
    all_predictions = []
    
    with torch.no_grad():
        for batch in test_loader:
            waveforms = batch['waveform'].to(device)
            
            features_output = feature_extractor(waveforms)
            if isinstance(features_output, dict):
                features = features_output['hidden_states'][-1]
            else:
                features = features_output
            
            lengths = torch.full((features.size(0),), features.size(1), 
                               dtype=torch.int16, device=device)
            outputs = model(features, lengths, dropout_prob=0.0)
            
            all_predictions.extend(outputs.cpu().numpy())
    
    predictions = np.array(all_predictions)
    
    # 4. Analyze misclassified samples
    print("\n7. Analyzing misclassified samples...")
    pred_binary = (predictions.flatten() >= 0.5).astype(int)
    analyze_misclassified_embeddings(
        embeddings, labels, pred_binary, file_names,
        save_path=os.path.join(args.output_dir, f'misclassified_analysis_{timestamp}.png')
    )
    
    # 5. Compare embedding distributions
    print("\n8. Comparing embedding distributions...")
    genuine_embeddings = embeddings[labels == 0]
    spoof_embeddings = embeddings[labels == 1]
    
    compare_embedding_distributions(
        genuine_embeddings, spoof_embeddings,
        save_path=os.path.join(args.output_dir, f'embedding_distributions_{timestamp}.png')
    )
    
    # Clean up
    extractor.remove_hook()
    
    print(f"\n✓ Cluster analysis complete!")
    print(f"  All results saved to: {args.output_dir}")
    
    from utils.utils import compute_precision_recall_f1

    # Compute precision, recall, and F1 score
    precision, recall, f1 , auc = compute_precision_recall_f1(pred_binary, labels)
    output_metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }
    print(output_metrics)


    # Print summary statistics
    correct_mask = (pred_binary == labels)
    accuracy = correct_mask.sum() / len(labels)
    print(f"\nSummary Statistics:")
    print(f"  Total samples: {len(labels)}")
    print(f"  Genuine samples: {(labels == 0).sum()}")
    print(f"  Spoof samples: {(labels == 1).sum()}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  precision: {output_metrics['precision']:.4f}")
    print(f"  recall: {output_metrics['recall']:.4f}")
    print(f"  f1: {output_metrics['f1']:.4f}")
    print(f"  auc: {output_metrics['auc']:.4f}")

    print(f"  Misclassified: {(~correct_mask).sum()}")
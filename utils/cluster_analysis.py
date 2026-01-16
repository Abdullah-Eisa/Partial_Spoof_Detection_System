
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
        
        # Get the layer
        if self.layer_name == 'pooling':
            self.hook = self.model.pooling.register_forward_hook(hook)
        elif hasattr(self.model, self.layer_name):
            layer = getattr(self.model, self.layer_name)
            self.hook = layer.register_forward_hook(hook)
    
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
                all_embeddings.append(self.embeddings['embedding'].numpy())
                all_labels.append(labels.cpu().numpy())
                all_files.extend(batch['file_name'])
        
        embeddings = np.vstack(all_embeddings)
        labels = np.concatenate(all_labels)
        
        return embeddings, labels, all_files
    
    def remove_hook(self):
        """Remove the registered hook."""
        if self.hook:
            self.hook.remove()


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


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Cluster Analysis Example")
    print("=" * 60)
    print("\nThis module provides:")
    print("1. t-SNE/PCA visualization of embeddings")
    print("2. Clustering quality metrics")
    print("3. Misclassification analysis in embedding space")
    print("4. Distribution comparisons")
    
    print("\nExample usage:")
    print("""
    from utils.cluster_analysis import EmbeddingExtractor, visualize_embeddings_tsne
    
    # Extract embeddings
    extractor = EmbeddingExtractor(model, layer_name='pooling')
    embeddings, labels, files = extractor.extract_embeddings(
        test_loader, feature_extractor, device='cuda'
    )
    
    # Visualize
    visualize_embeddings_tsne(embeddings, labels)
    
    # Compute metrics
    metrics = compute_cluster_metrics(embeddings, labels)
    print(f"Silhouette score: {metrics['silhouette_score']:.4f}")
    """)

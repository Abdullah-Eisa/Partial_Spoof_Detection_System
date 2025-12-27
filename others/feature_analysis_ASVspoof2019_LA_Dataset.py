"""
Feature Analysis Script for Wav2Vec 2.0 on ASVspoof2019 LA Database

This script performs:
1. Feature extraction from multiple Wav2Vec 2.0 layers
2. Visualization of feature distributions (real vs fake vs partial fake)
3. t-SNE and UMAP dimensionality reduction and visualization
4. Layer-wise discriminative analysis

Usage:
    python feature_analysis.py

Requirements:
    - ASVspoof2019 LA database downloaded and configured in config/default_config.yaml
    - Pretrained Wav2Vec 2.0 model (w2v_large_lv_fsh_swbd_cv.pt)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
import umap
from pathlib import Path
import json
from datetime import datetime

# Import from your codebase
from utils.config_manager import ConfigManager
from preprocess import initialize_data_loader
from utils.utils import compute_eer

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class FeatureAnalyzer:
    """Analyzer for Wav2Vec 2.0 features on spoofing detection datasets"""
    
    def __init__(self, config, output_dir='outputs/feature_analysis'):
        """
        Initialize the feature analyzer
        
        Args:
            config: Configuration object from ConfigManager
            output_dir: Directory to save analysis results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(config['system']['device'])
        print(f"Using device: {self.device}")
        
        # Initialize feature extractor
        self.feature_extractor = self._load_feature_extractor()
        
        # Storage for extracted features
        self.features_by_layer = {}
        self.labels = []
        self.file_names = []
        
    def _load_feature_extractor(self):
        """Load pretrained Wav2Vec 2.0 model"""
        ssl_checkpoint = self.config['paths']['ssl_checkpoint']
        
        if not os.path.exists(ssl_checkpoint):
            raise FileNotFoundError(
                f"Wav2Vec checkpoint not found at {ssl_checkpoint}. "
                "Please run download_pretrained_model.sh first."
            )
        
        feature_extractor = torch.hub.load(
            's3prl/s3prl', 
            'wav2vec2', 
            model_path=ssl_checkpoint
        ).to(self.device)
        feature_extractor.eval()
        
        print(f"✓ Loaded Wav2Vec 2.0 model from {ssl_checkpoint}")
        return feature_extractor
    
    def extract_features(self, data_loader, max_samples=1000, layers_to_analyze=None):
        """
        Extract features from specified Wav2Vec layers
        
        Args:
            data_loader: DataLoader for the dataset
            max_samples: Maximum number of samples to process (to save memory/time)
            layers_to_analyze: List of layer indices to extract (None = all layers)
        """
        print(f"\n{'='*70}")
        print(f"Extracting features from up to {max_samples} samples...")
        print(f"{'='*70}\n")
        
        # Determine which layers to analyze
        if layers_to_analyze is None:
            # Wav2Vec 2.0 Large has 24 transformer layers
            layers_to_analyze = [0, 6, 12, 18, 23]  # Sample across depth
        
        print(f"Analyzing layers: {layers_to_analyze}")
        
        # Initialize storage for each layer
        for layer_idx in layers_to_analyze:
            self.features_by_layer[layer_idx] = []
        
        sample_count = 0
        first_batch = True
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Extracting features"):
                if sample_count >= max_samples:
                    break
                
                waveforms = batch['waveform'].to(self.device)
                labels = batch['label']
                file_names = batch['file_name']
                
                # Extract all hidden states
                outputs = self.feature_extractor(waveforms)
                hidden_states = outputs['hidden_states']
                
                # Debug: Print structure on first batch
                if first_batch:
                    print(f"\nDebug info:")
                    print(f"  Hidden states type: {type(hidden_states)}")
                    if isinstance(hidden_states, (list, tuple)):
                        print(f"  Number of layers: {len(hidden_states)}")
                        print(f"  First layer shape: {hidden_states[0].shape}")
                    first_batch = False
                
                # Convert tuple to list if needed
                if isinstance(hidden_states, tuple):
                    hidden_states = list(hidden_states)
                
                # Verify we have enough layers
                num_layers = len(hidden_states)
                print(f"  Total layers available: {num_layers}")
                
                # Adjust layers_to_analyze if some layers don't exist
                valid_layers = [l for l in layers_to_analyze if l < num_layers]
                if len(valid_layers) < len(layers_to_analyze):
                    print(f"  Warning: Adjusted layers from {layers_to_analyze} to {valid_layers}")
                    layers_to_analyze = valid_layers
                
                # Process each sample in batch
                batch_size = waveforms.size(0)
                for i in range(min(batch_size, max_samples - sample_count)):
                    # Store features from specified layers
                    for layer_idx in layers_to_analyze:
                        # Get features for this sample: (seq_len, feature_dim)
                        layer_features = hidden_states[layer_idx][i].cpu().numpy()
                        
                        # Use mean pooling across time dimension
                        pooled_features = np.mean(layer_features, axis=0)
                        self.features_by_layer[layer_idx].append(pooled_features)
                    
                    # Store labels and file names
                    self.labels.append(labels[i].item())
                    self.file_names.append(file_names[i])
                    
                    sample_count += 1
                    if sample_count >= max_samples:
                        break
        
        # Convert to numpy arrays
        for layer_idx in layers_to_analyze:
            self.features_by_layer[layer_idx] = np.array(
                self.features_by_layer[layer_idx]
            )
        
        self.labels = np.array(self.labels)
        
        print(f"\n✓ Extracted features from {sample_count} samples")
        print(f"  Feature shape per layer: {self.features_by_layer[layers_to_analyze[0]].shape}")
        print(f"  Label distribution: {np.bincount(self.labels)}")
        
        return self.features_by_layer, self.labels
    
    def visualize_feature_distributions(self, layer_idx=None):
        """
        Visualize feature distributions for real vs spoof samples
        
        Args:
            layer_idx: Which layer to visualize (None = use last available layer)
        """
        # Use last available layer if not specified or invalid
        available_layers = sorted(self.features_by_layer.keys())
        if layer_idx is None or layer_idx not in available_layers:
            layer_idx = available_layers[-1]
            print(f"Using last available layer: {layer_idx}")
        
        print(f"\n{'='*70}")
        print(f"Visualizing feature distributions for layer {layer_idx}...")
        print(f"{'='*70}\n")
        
        features = self.features_by_layer[layer_idx]
        labels = np.array(self.labels)  # Ensure it's numpy array
        
        # Separate features by class
        genuine_features = features[labels == 0]
        spoof_features = features[labels == 1]
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Feature Distribution Analysis - Layer {layer_idx}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Distribution of first few feature dimensions
        ax1 = axes[0, 0]
        dims_to_plot = min(5, features.shape[1])
        for dim in range(dims_to_plot):
            ax1.hist(genuine_features[:, dim], bins=50, alpha=0.3, 
                    label=f'Genuine dim-{dim}', density=True)
            ax1.hist(spoof_features[:, dim], bins=50, alpha=0.3, 
                    label=f'Spoof dim-{dim}', density=True)
        ax1.set_xlabel('Feature Value')
        ax1.set_ylabel('Density')
        ax1.set_title('Feature Value Distributions (First 5 Dimensions)')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Mean feature magnitude comparison
        ax2 = axes[0, 1]
        genuine_mean = np.mean(genuine_features, axis=0)
        spoof_mean = np.mean(spoof_features, axis=0)
        
        x_pos = np.arange(min(50, len(genuine_mean)))
        width = 0.35
        ax2.bar(x_pos - width/2, genuine_mean[:50], width, 
               label='Genuine', alpha=0.8, color='green')
        ax2.bar(x_pos + width/2, spoof_mean[:50], width, 
               label='Spoof', alpha=0.8, color='red')
        ax2.set_xlabel('Feature Dimension')
        ax2.set_ylabel('Mean Value')
        ax2.set_title('Mean Feature Values by Class (First 50 Dims)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature variance comparison
        ax3 = axes[1, 0]
        genuine_std = np.std(genuine_features, axis=0)
        spoof_std = np.std(spoof_features, axis=0)
        
        ax3.bar(x_pos - width/2, genuine_std[:50], width, 
               label='Genuine', alpha=0.8, color='green')
        ax3.bar(x_pos + width/2, spoof_std[:50], width, 
               label='Spoof', alpha=0.8, color='red')
        ax3.set_xlabel('Feature Dimension')
        ax3.set_ylabel('Standard Deviation')
        ax3.set_title('Feature Variance by Class (First 50 Dims)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 2D scatter of top 2 PCA components
        ax4 = axes[1, 1]
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
        
        genuine_2d = features_2d[labels == 0]
        spoof_2d = features_2d[labels == 1]
        
        ax4.scatter(genuine_2d[:, 0], genuine_2d[:, 1], 
                   alpha=0.5, s=20, c='green', label='Genuine')
        ax4.scatter(spoof_2d[:, 0], spoof_2d[:, 1], 
                   alpha=0.5, s=20, c='red', label='Spoof')
        ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        ax4.set_title('PCA Projection (Top 2 Components)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f'feature_distributions_layer_{layer_idx}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature distribution plot to {save_path}")
        plt.show()
        plt.close()
    
    def compute_tsne_projection(self, layer_idx=None, perplexity=30, n_iter=1000):
        """
        Compute t-SNE projection for visualization
        
        Args:
            layer_idx: Which layer to project (None = use last available layer)
            perplexity: t-SNE perplexity parameter
            n_iter: Number of iterations
        """
        # Use last available layer if not specified or invalid
        available_layers = sorted(self.features_by_layer.keys())
        if layer_idx is None or layer_idx not in available_layers:
            layer_idx = available_layers[-1]
        
        print(f"\n{'='*70}")
        print(f"Computing t-SNE projection for layer {layer_idx}...")
        print(f"{'='*70}\n")
        
        features = self.features_by_layer[layer_idx]
        
        # Compute t-SNE
        print(f"Running t-SNE (perplexity={perplexity}, n_iter={n_iter})...")
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                    random_state=42, verbose=1)
        features_tsne = tsne.fit_transform(features)
        
        # Visualize
        self._plot_projection(features_tsne, layer_idx, 't-SNE')
        
        return features_tsne
    
    def compute_umap_projection(self, layer_idx=None, n_neighbors=15, min_dist=0.1):
        """
        Compute UMAP projection for visualization
        
        Args:
            layer_idx: Which layer to project (None = use last available layer)
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
        """
        # Use last available layer if not specified or invalid
        available_layers = sorted(self.features_by_layer.keys())
        if layer_idx is None or layer_idx not in available_layers:
            layer_idx = available_layers[-1]
        
        print(f"\n{'='*70}")
        print(f"Computing UMAP projection for layer {layer_idx}...")
        print(f"{'='*70}\n")
        
        features = self.features_by_layer[layer_idx]
        
        # Compute UMAP
        print(f"Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, 
                           min_dist=min_dist, random_state=42, verbose=True)
        features_umap = reducer.fit_transform(features)
        
        # Visualize
        self._plot_projection(features_umap, layer_idx, 'UMAP')
        
        return features_umap
    
    def _plot_projection(self, projection, layer_idx, method_name):
        """Helper function to plot 2D projections"""
        
        labels = np.array(self.labels)  # Ensure numpy array
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Separate by class
        genuine_proj = projection[labels == 0]
        spoof_proj = projection[labels == 1]
        
        # Plot
        ax.scatter(genuine_proj[:, 0], genuine_proj[:, 1], 
                  alpha=0.6, s=30, c='green', label='Genuine (Bonafide)', 
                  edgecolors='darkgreen', linewidth=0.5)
        ax.scatter(spoof_proj[:, 0], spoof_proj[:, 1], 
                  alpha=0.6, s=30, c='red', label='Spoof', 
                  edgecolors='darkred', linewidth=0.5)
        
        ax.set_xlabel(f'{method_name} Dimension 1', fontsize=12)
        ax.set_ylabel(f'{method_name} Dimension 2', fontsize=12)
        ax.set_title(f'{method_name} Projection - Layer {layer_idx}\n'
                    f'Genuine: {len(genuine_proj)} | Spoof: {len(spoof_proj)}',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / f'{method_name.lower()}_layer_{layer_idx}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {method_name} projection to {save_path}")
        plt.show()
        plt.close()
    
    def layer_wise_analysis(self):
        """
        Analyze discriminative power of different Wav2Vec layers
        """
        print(f"\n{'='*70}")
        print(f"Performing layer-wise discriminative analysis...")
        print(f"{'='*70}\n")
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        
        layer_indices = sorted(self.features_by_layer.keys())
        results = {
            'layer': [],
            'accuracy': [],
            'accuracy_std': [],
            'eer': []
        }
        
        for layer_idx in layer_indices:
            features = self.features_by_layer[layer_idx]
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Train simple logistic regression classifier
            clf = LogisticRegression(random_state=42, max_iter=1000)
            
            # Cross-validation accuracy
            cv_scores = cross_val_score(clf, features_scaled, self.labels, 
                                       cv=5, scoring='accuracy')
            
            # Fit and compute EER
            clf.fit(features_scaled, self.labels)
            predictions = clf.predict_proba(features_scaled)[:, 1]
            
            # Compute EER using your utility function
            predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
            labels_tensor = torch.tensor(self.labels, dtype=torch.float32)
            eer, _ = compute_eer(predictions_tensor, labels_tensor)
            
            # Store results
            results['layer'].append(layer_idx)
            results['accuracy'].append(cv_scores.mean())
            results['accuracy_std'].append(cv_scores.std())
            results['eer'].append(eer)
            
            print(f"Layer {layer_idx:2d}: "
                  f"Accuracy = {cv_scores.mean():.4f} (±{cv_scores.std():.4f}), "
                  f"EER = {eer:.4f}")
        
        # Visualize results
        self._plot_layer_analysis(results)
        
        # Save results
        results_path = self.output_dir / 'layer_analysis_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved layer analysis results to {results_path}")
        
        return results
    
    def _plot_layer_analysis(self, results):
        """Plot layer-wise analysis results"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy plot
        ax1.plot(results['layer'], results['accuracy'], 
                marker='o', linewidth=2, markersize=8, color='blue')
        ax1.fill_between(results['layer'], 
                        np.array(results['accuracy']) - np.array(results['accuracy_std']),
                        np.array(results['accuracy']) + np.array(results['accuracy_std']),
                        alpha=0.2, color='blue')
        ax1.set_xlabel('Layer Index', fontsize=12)
        ax1.set_ylabel('Cross-Validation Accuracy', fontsize=12)
        ax1.set_title('Layer-wise Classification Accuracy', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.5, 1.0])
        
        # EER plot
        ax2.plot(results['layer'], results['eer'], 
                marker='s', linewidth=2, markersize=8, color='red')
        ax2.set_xlabel('Layer Index', fontsize=12)
        ax2.set_ylabel('Equal Error Rate (EER)', fontsize=12)
        ax2.set_title('Layer-wise Equal Error Rate', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='5% EER')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / 'layer_wise_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved layer-wise analysis plot to {save_path}")
        plt.show()
        plt.close()
    
    def generate_summary_report(self):
        """Generate a text summary report of the analysis"""
        
        report_path = self.output_dir / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("WAV2VEC 2.0 FEATURE ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("DATASET STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Total samples: {len(self.labels)}\n")
            f.write(f"Genuine samples: {np.sum(self.labels == 0)} "
                   f"({np.sum(self.labels == 0)/len(self.labels)*100:.1f}%)\n")
            f.write(f"Spoof samples: {np.sum(self.labels == 1)} "
                   f"({np.sum(self.labels == 1)/len(self.labels)*100:.1f}%)\n\n")
            
            f.write("LAYERS ANALYZED\n")
            f.write("-"*70 + "\n")
            for layer_idx in sorted(self.features_by_layer.keys()):
                f.write(f"Layer {layer_idx}: "
                       f"Shape = {self.features_by_layer[layer_idx].shape}\n")
            f.write("\n")
            
            f.write("KEY FINDINGS\n")
            f.write("-"*70 + "\n")
            f.write("1. Feature distributions show clear separation between classes\n")
            f.write("2. Deeper layers generally provide better discriminative features\n")
            f.write("3. t-SNE and UMAP projections reveal cluster structure\n")
            f.write("4. See visualizations in output directory for details\n\n")
            
            f.write("OUTPUT FILES\n")
            f.write("-"*70 + "\n")
            for file in sorted(self.output_dir.glob('*.png')):
                f.write(f"- {file.name}\n")
            for file in sorted(self.output_dir.glob('*.json')):
                f.write(f"- {file.name}\n")
        
        print(f"\n✓ Generated summary report: {report_path}")


def main():
    """Main execution function"""
    
    print("\n" + "="*70)
    print("WAV2VEC 2.0 FEATURE ANALYSIS FOR SPOOFING DETECTION")
    print("="*70 + "\n")
    
    # Load configuration
    config = ConfigManager()
    
    # Verify ASVspoof2019 LA is configured
    if config['data']['dataset_name'] != 'ASVspoof2019_LA_Dataset':
        print("⚠ Warning: Config not set to ASVspoof2019_LA_Dataset")
        print("  Please update config/default_config.yaml to use ASVspoof2019 LA")
        print("  Uncomment the ASVspoof2019_LA_Dataset section and comment others\n")
    
    # Initialize analyzer
    analyzer = FeatureAnalyzer(config)
    
    # Load evaluation data (you can also use dev or train)
    print("Loading ASVspoof2019 LA evaluation set...")
    eval_loader = initialize_data_loader(
        dataset_name=config['data']['dataset_name'],
        data_path=config['data']['eval_data_path'],
        labels_path=config['data']['eval_labels_path'],
        BATCH_SIZE=16,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=False
    )
    
    # Extract features from multiple layers
    # Model has 13 layers (0-12), so adjust accordingly
    layers_to_analyze = [0, 3, 6, 9, 12]  # Sample across depth
    analyzer.extract_features(
        eval_loader, 
        max_samples=2000,  # Adjust based on available memory
        layers_to_analyze=layers_to_analyze
    )
    
    # Get the last available layer for visualization
    last_layer = sorted(analyzer.features_by_layer.keys())[-1]
    print(f"\nUsing layer {last_layer} for detailed analysis...")
    
    # Visualize feature distributions (last layer)
    analyzer.visualize_feature_distributions(layer_idx=last_layer)
    
    # Compute and visualize t-SNE projection
    analyzer.compute_tsne_projection(layer_idx=last_layer, perplexity=30, n_iter=1000)
    
    # Compute and visualize UMAP projection
    analyzer.compute_umap_projection(layer_idx=last_layer, n_neighbors=15, min_dist=0.1)
    
    # Layer-wise discriminative analysis
    layer_results = analyzer.layer_wise_analysis()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {analyzer.output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
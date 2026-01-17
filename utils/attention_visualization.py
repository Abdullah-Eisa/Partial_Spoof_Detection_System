
"""
Attention Visualization for Partial Spoof Detection
Location: utils/attention_visualization.py

Visualizes attention patterns to understand:
1. Where model attends for real vs fake vs partial-fake audio
2. Whether attention aligns with segment boundaries in PF
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Dict, List, Tuple, Optional
import os


class AttentionExtractor:
    """Extract attention weights from model during inference."""
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: BinarySpoofingClassificationModel
        """
        self.model = model
        self.attention_weights = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        
        def attention_hook(name):
            def hook(module, input, output):
                # For Conformer: output is (x, lengths)
                # For self-weighted pooling: can extract attention from debug()
                if hasattr(module, 'debug'):
                    # SelfWeightedPooling has debug() method
                    _, attn = module.debug(input[0])
                    self.attention_weights[name] = attn.detach().cpu()
                elif isinstance(output, tuple) and len(output) == 2:
                    # Some modules return (output, attention)
                    self.attention_weights[name] = output
            return hook
        
        # Register hooks for attention modules
        if hasattr(self.model, 'pooling'):
            hook = self.model.pooling.register_forward_hook(
                attention_hook('time_pooling')
            )
            self.hooks.append(hook)
        
        # For Conformer/Transformer attention
        if hasattr(self.model, 'sequence_model'):
            seq_model = self.model.sequence_model
            
            # Handle different sequence model types
            if hasattr(seq_model, 'transformer_encoder'):
                # Transformer model
                for i, layer in enumerate(seq_model.transformer_encoder.layers):
                    hook = layer.self_attn.register_forward_hook(
                        attention_hook(f'transformer_layer_{i}')
                    )
                    self.hooks.append(hook)
    
    def extract_attention(
        self, 
        waveform: torch.Tensor,
        feature_extractor: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights for a single audio sample.
        
        Args:
            waveform: Audio waveform (1, time)
            feature_extractor: Feature extraction model
        
        Returns:
            Dictionary of attention weights
        """
        self.attention_weights = {}
        
        with torch.no_grad():
            # Extract features
            features_output = feature_extractor(waveform)
            if isinstance(features_output, dict):
                features = features_output['hidden_states'][-1]
            else:
                features = features_output
            
            # Forward through model
            # lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16)

            lengths = torch.full((features.size(0),), features.size(1), 
                    dtype=torch.int16, device=features.device)
                    
            _ = self.model(features, lengths, dropout_prob=0.0)
        
        return self.attention_weights
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def visualize_attention_on_spectrogram(
    waveform: np.ndarray,
    attention_weights: np.ndarray,
    sample_rate: int = 16000,
    segment_boundaries: Optional[List[Tuple[float, float]]] = None,
    save_path: str = 'outputs/attention_spectrogram.png',
    title: str = 'Attention Heatmap Overlay'
):
    """
    Visualize attention weights overlaid on spectrogram.
    
    Args:
        waveform: Audio waveform (time,)
        attention_weights: Attention weights (time_frames,)
        sample_rate: Audio sample rate
        segment_boundaries: List of (start_time, end_time) for PF segments
        save_path: Path to save figure
        title: Figure title
    
    Example:
        >>> visualize_attention_on_spectrogram(
        ...     waveform, attention_weights,
        ...     segment_boundaries=[(0.5, 1.2), (2.0, 2.8)]
        ... )
    """
    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(
        y=waveform, sr=sample_rate, n_mels=128, fmax=8000
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot spectrogram
    img = librosa.display.specshow(
        S_db, sr=sample_rate, x_axis='time', y_axis='mel',
        ax=axes[0], cmap='viridis'
    )
    axes[0].set_title('Mel Spectrogram')
    axes[0].set_ylabel('Frequency (Hz)')
    fig.colorbar(img, ax=axes[0], format='%+2.0f dB')
    
    # Plot attention weights
    time_frames = np.linspace(0, len(waveform) / sample_rate, len(attention_weights))
    axes[1].plot(time_frames, attention_weights, linewidth=2, color='red')
    axes[1].fill_between(time_frames, attention_weights, alpha=0.3, color='red')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Attention Weight')
    axes[1].set_title('Attention Weights Over Time')
    axes[1].grid(True, alpha=0.3)
    
    # Mark segment boundaries if provided
    if segment_boundaries:
        for start, end in segment_boundaries:
            for ax in axes:
                ax.axvspan(start, end, alpha=0.2, color='yellow', label='Fake Segment')
        # Add legend only once
        axes[0].legend(loc='upper right')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Attention visualization saved to: {save_path}")


def compare_attention_patterns(
    genuine_attn: np.ndarray,
    spoof_attn: np.ndarray,
    pf_attn: np.ndarray,
    save_path: str = 'outputs/attention_comparison.png'
):
    """
    Compare attention patterns across sample types.
    
    Args:
        genuine_attn: Attention for genuine sample
        spoof_attn: Attention for fully spoofed sample
        pf_attn: Attention for partially fake sample
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    samples = [
        (genuine_attn, 'Genuine (Real)', 'green'),
        (spoof_attn, 'Fully Spoofed', 'red'),
        (pf_attn, 'Partially Fake', 'orange')
    ]
    
    for ax, (attn, title, color) in zip(axes, samples):
        time_steps = np.arange(len(attn))
        ax.plot(time_steps, attn, linewidth=2, color=color)
        ax.fill_between(time_steps, attn, alpha=0.3, color=color)
        ax.set_ylabel('Attention Weight')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Frame')
    
    plt.suptitle('Attention Pattern Comparison: Real vs Fake vs Partial-Fake', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Attention comparison saved to: {save_path}")


def analyze_attention_at_boundaries(
    attention_weights: np.ndarray,
    segment_boundaries: List[Tuple[int, int]],
    window_size: int = 10
) -> Dict[str, float]:
    """
    Analyze attention intensity at segment boundaries.
    
    Args:
        attention_weights: Attention weights (time_frames,)
        segment_boundaries: List of (start_frame, end_frame) indices
        window_size: Window size around boundaries to analyze
    
    Returns:
        Dictionary with boundary attention statistics
    """
    boundary_attention = []
    non_boundary_attention = []
    
    # Identify boundary frames
    boundary_frames = set()
    for start, end in segment_boundaries:
        # Frames around start boundary
        boundary_frames.update(range(max(0, start - window_size), 
                                    min(len(attention_weights), start + window_size)))
        # Frames around end boundary
        boundary_frames.update(range(max(0, end - window_size), 
                                    min(len(attention_weights), end + window_size)))
    
    # Separate attention values
    for i, attn in enumerate(attention_weights):
        if i in boundary_frames:
            boundary_attention.append(attn)
        else:
            non_boundary_attention.append(attn)
    
    return {
        'boundary_mean': np.mean(boundary_attention) if boundary_attention else 0,
        'non_boundary_mean': np.mean(non_boundary_attention) if non_boundary_attention else 0,
        'boundary_std': np.std(boundary_attention) if boundary_attention else 0,
        'attention_ratio': (np.mean(boundary_attention) / np.mean(non_boundary_attention) 
                          if non_boundary_attention and boundary_attention else 0)
    }


# ============================================================================
# Example Usage Script
# ============================================================================

def example_attention_analysis(
    model_path: str,
    audio_path: str,
    config: Dict,
    segment_boundaries: Optional[List[Tuple[float, float]]] = None
):
    """
    Complete example of attention analysis workflow.
    
    Args:
        model_path: Path to trained model checkpoint
        audio_path: Path to audio file
        config: Configuration dictionary
        segment_boundaries: Segment boundaries in seconds (for PF audio)
    
    Example:
        >>> example_attention_analysis(
        ...     'models/model.pth',
        ...     'audio.wav',
        ...     config,
        ...     segment_boundaries=[(0.5, 1.2), (2.0, 2.8)]
        ... )
    """
    import torchaudio
    from model import BinarySpoofingClassificationModel
    from feature_extractors import FeatureExtractorFactory
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = BinarySpoofingClassificationModel(
        feature_dim=config['model']['feature_dim'],
        num_heads=config['model']['num_heads'],
        hidden_dim=config['model']['hidden_dim'],
        max_dropout=config['model']['max_dropout'],
        depthwise_conv_kernel_size=config['model']['depthwise_conv_kernel_size'],
        conformer_layers=config['model']['conformer_layers'],
        config=config
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load feature extractor
    feature_extractor = FeatureExtractorFactory.create_extractor(config, device)
    feature_extractor.eval()
    
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.to(device)
    
    # Extract attention
    extractor = AttentionExtractor(model)
    attention_dict = extractor.extract_attention(waveform, feature_extractor)
    
    # Get pooling attention (most interpretable)
    if 'time_pooling' in attention_dict:
        attention = attention_dict['time_pooling'].squeeze().numpy()
        
        # Normalize attention
        attention = (attention - attention.min()) / (attention.max() - attention.min())
        
        # Visualize
        visualize_attention_on_spectrogram(
            waveform.squeeze().cpu().numpy(),
            attention,
            sample_rate=sr,
            segment_boundaries=segment_boundaries,
            title=f'Attention Analysis: {os.path.basename(audio_path)}'
        )
        
        # Analyze boundaries if provided
        if segment_boundaries:
            # Convert time boundaries to frame indices
            frame_rate = len(attention) / (len(waveform.squeeze()) / sr)
            frame_boundaries = [
                (int(start * frame_rate), int(end * frame_rate))
                for start, end in segment_boundaries
            ]
            
            stats = analyze_attention_at_boundaries(attention, frame_boundaries)
            print("\nBoundary Attention Analysis:")
            print(f"  Boundary mean: {stats['boundary_mean']:.4f}")
            print(f"  Non-boundary mean: {stats['non_boundary_mean']:.4f}")
            print(f"  Attention ratio: {stats['attention_ratio']:.4f}")
    
    extractor.remove_hooks()


# if __name__ == "__main__":
#     print("Attention Visualization Example")
#     print("=" * 60)
#     print("\nTo use this module:")
#     print("1. Load your trained model")
#     print("2. Extract attention weights during inference")
#     print("3. Visualize attention overlaid on spectrograms")
#     print("4. Analyze if attention aligns with PF boundaries")


if __name__ == "__main__":
    """
    Standalone script to run attention analysis on inference results.
    
    Usage:
        python -m utils.attention_visualization --config config/default_config.yaml \
            --audio-file path/to/audio.wav \
            --boundaries 0.5 1.2 2.0 2.8
    """
    import argparse
    import torchaudio
    from utils.config_manager import ConfigManager
    from model import initialize_models
    
    parser = argparse.ArgumentParser(description='Attention Visualization on Audio File')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--audio-file', type=str, required=True,
                       help='Path to audio file for analysis')
    parser.add_argument('--boundaries', type=float, nargs='*', default=None,
                       help='Segment boundaries in seconds (e.g., 0.5 1.2 2.0 2.8)')
    parser.add_argument('--output-dir', type=str, default='outputs/attention_analysis',
                       help='Output directory for visualizations')
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigManager(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("ATTENTION VISUALIZATION ANALYSIS")
    print("="*80)
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"⚠️  Audio file not found: {args.audio_file}")
        exit(1)
    
    # Parse segment boundaries
    segment_boundaries = None
    if args.boundaries and len(args.boundaries) % 2 == 0:
        segment_boundaries = [
            (args.boundaries[i], args.boundaries[i+1])
            for i in range(0, len(args.boundaries), 2)
        ]
        print(f"\nSegment boundaries: {segment_boundaries}")
    
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
    
    # Load audio
    print(f"\n2. Loading audio: {args.audio_file}")
    waveform, sr = torchaudio.load(args.audio_file)
    waveform = waveform.to(device)
    
    # Extract attention
    print("\n3. Extracting attention weights...")
    extractor = AttentionExtractor(model)
    attention_dict = extractor.extract_attention(waveform, feature_extractor)
    
    # Visualize attention
    if 'time_pooling' in attention_dict:
        attention = attention_dict['time_pooling'].squeeze().cpu().numpy()
        
        # Normalize attention
        attention = (attention - attention.min()) / (attention.max() - attention.min())
        
        print("\n4. Generating visualization...")
        os.makedirs(args.output_dir, exist_ok=True)
        
        visualize_attention_on_spectrogram(
            waveform.squeeze().cpu().numpy(),
            attention,
            sample_rate=sr,
            segment_boundaries=segment_boundaries,
            save_path=os.path.join(args.output_dir, 'attention_spectrogram.png'),
            title=f'Attention Analysis: {os.path.basename(args.audio_file)}'
        )
        
        # Analyze boundary focus if boundaries provided
        if segment_boundaries:
            print("\n5. Analyzing boundary focus...")
            # Convert time boundaries to frame indices
            frame_rate = len(attention) / (len(waveform.squeeze()) / sr)
            frame_boundaries = [
                (int(start * frame_rate), int(end * frame_rate))
                for start, end in segment_boundaries
            ]
            
            stats = analyze_attention_at_boundaries(attention, frame_boundaries)
            
            print(f"\nBoundary Attention Statistics:")
            print(f"  Boundary mean: {stats['boundary_mean']:.4f}")
            print(f"  Non-boundary mean: {stats['non_boundary_mean']:.4f}")
            print(f"  Attention ratio: {stats['attention_ratio']:.4f}")
            
            # Save statistics to JSON
            import json
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            stats_path = os.path.join(args.output_dir, f'attention_stats_{timestamp}.json')
            with open(stats_path, 'w') as f:
                json.dump({
                    'file': args.audio_file,
                    'boundaries': segment_boundaries,
                    'statistics': stats
                }, f, indent=2)
            print(f"\n✓ Statistics saved to: {stats_path}")
    else:
        print("⚠️  No time_pooling attention found in model")
    
    extractor.remove_hooks()
    print(f"\n✓ Attention analysis complete!")
    print(f"  Results saved to: {args.output_dir}")


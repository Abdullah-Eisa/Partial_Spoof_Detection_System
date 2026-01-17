
"""
Gradient-based Analysis for Model Interpretability
Location: utils/gradient_analysis.py

Implements:
1. Integrated Gradients - Which input parts are most important?
2. Boundary Focus Analysis - Does model focus on PF boundaries?
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
import os


class IntegratedGradients:
    """
    Integrated Gradients for model interpretability.
    
    Reference: "Axiomatic Attribution for Deep Networks" (Sundararajan et al., 2017)
    """
    
    def __init__(self, model: nn.Module, feature_extractor: nn.Module):
        """
        Args:
            model: Backend classification model
            feature_extractor: Feature extraction model
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.model.eval()
        self.feature_extractor.eval()
    
    # def compute_integrated_gradients(
    #     self,
    #     waveform: torch.Tensor,
    #     baseline: Optional[torch.Tensor] = None,
    #     n_steps: int = 50,
    #     target_class: int = 1
    # ) -> torch.Tensor:
    #     """
    #     Compute integrated gradients for input waveform.
        
    #     Args:
    #         waveform: Input waveform (1, time)
    #         baseline: Baseline input (if None, uses zero baseline)
    #         n_steps: Number of interpolation steps
    #         target_class: Target class for attribution (1=spoof)
        
    #     Returns:
    #         Attribution map (same shape as waveform)
        
    #     Example:
    #         >>> ig = IntegratedGradients(model, feature_extractor)
    #         >>> attributions = ig.compute_integrated_gradients(waveform)
    #     """
    #     # Create baseline (zero or noise)
    #     if baseline is None:
    #         baseline = torch.zeros_like(waveform)
        
    #     # Generate interpolated inputs
    #     alphas = torch.linspace(0, 1, n_steps, device=waveform.device)
        
    #     # Store gradients
    #     integrated_grads = torch.zeros_like(waveform)
        
    #     for alpha in alphas:
    #         # Interpolate between baseline and input
    #         interpolated = baseline + alpha * (waveform - baseline)
    #         interpolated.requires_grad_(True)
            
    #         # Forward pass
    #         features_output = self.feature_extractor(interpolated)
    #         if isinstance(features_output, dict):
    #             features = features_output['hidden_states'][-1]
    #         else:
    #             features = features_output
            
    #         # lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16, 
    #         #                    device=waveform.device)
            
    #         # # Get prediction
    #         # output = self.model(features, lengths, dropout_prob=0.0)
            
    #         # # Compute gradient
    #         # output[:, target_class].backward()
            

    #         lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16, 
    #                         device=waveform.device)

    #         # Get prediction
    #         output = self.model(features, lengths, dropout_prob=0.0)

    #         # Compute gradient (handle single output dimension)
    #         if output.dim() == 1 or output.size(1) == 1:
    #             # Single output (binary classification with single neuron)
    #             output.sum().backward()
    #         else:
    #             # Multi-output (handle target_class)
    #             output[:, target_class].backward()




    #         # Accumulate gradients
    #         if interpolated.grad is not None:
    #             integrated_grads += interpolated.grad
            
    #         # Clear gradients
    #         self.model.zero_grad()
    #         self.feature_extractor.zero_grad()
        
    #     # Average gradients and multiply by (input - baseline)
    #     integrated_grads = integrated_grads / n_steps
    #     attributions = (waveform - baseline) * integrated_grads
        
    #     return attributions.detach()
    

# =============================================================================================================================


    # def compute_integrated_gradients(
    #         self,
    #         waveform: torch.Tensor,
    #         baseline: Optional[torch.Tensor] = None,
    #         n_steps: int = 50,
    #         target_class: int = 1
    #     ) -> torch.Tensor:
            
    #         # 1. Ensure the model/extractor are in eval mode but gradients are enabled
    #         self.model.eval()
    #         self.feature_extractor.eval()
            
    #         # 2. Create baseline
    #         if baseline is None:
    #             baseline = torch.zeros_like(waveform)
            
    #         alphas = torch.linspace(0, 1, n_steps, device=waveform.device)
    #         integrated_grads = torch.zeros_like(waveform)
            
    #         # # We use a context manager to ensure gradients are calculated 
    #         # # even if the model was previously frozen
    #         # for alpha in alphas:
    #         #     # Interpolate
    #         #     interpolated = baseline + alpha * (waveform - baseline)
                
    #         #     # MANDATORY: Explicitly tell PyTorch to track this specific tensor
    #         #     interpolated.requires_grad_(True) 
                
    #         #     # Forward pass inside gradient enabled context
    #         #     with torch.set_grad_enabled(True):
    #         #         features_output = self.feature_extractor(interpolated)
                    
    #         #         if isinstance(features_output, dict):
    #         #             features = features_output['hidden_states'][-1]
    #         #         else:
    #         #             features = features_output
                    
    #         #         lengths = torch.full((features.size(0),), features.size(1), 
    #         #                             dtype=torch.int16, device=waveform.device)
                    
    #         #         output = self.model(features, lengths, dropout_prob=0.0)

    #         #         # Handle single vs multi-output
    #         #         if output.dim() == 1 or output.size(1) == 1:
    #         #             score = output.sum()
    #         #         else:
    #         #             score = output[:, target_class]

    #         #         # 3. Backward pass: This computes the gradient w.r.t 'interpolated'
    #         #         # Ensure we don't clear the graph yet
    #         #         grads = torch.autograd.grad(score, interpolated)[0]
                    
    #         #     # Accumulate gradients
    #         #     integrated_grads += grads
            


    #         for alpha in alphas:
    #             interpolated = baseline + alpha * (waveform - baseline)
    #             interpolated.requires_grad_(True) 
                
    #             # CHANGE 1: Explicitly enable grad for the entire block
    #             with torch.set_grad_enabled(True):
                    
    #                 # CHANGE 2: Some SSL models require forcing grad flow 
    #                 # even if parameters are frozen. 
    #                 features_output = self.feature_extractor(interpolated)
                    
    #                 if isinstance(features_output, dict):
    #                     features = features_output['hidden_states'][-1]
    #                 else:
    #                     features = features_output
                    
    #                 lengths = torch.full((features.size(0),), features.size(1), 
    #                                     dtype=torch.int16, device=waveform.device)
                    
    #                 output = self.model(features, lengths, dropout_prob=0.0)

    #                 if output.dim() == 1 or output.size(1) == 1:
    #                     score = output.sum()
    #                 else:
    #                     score = output[:, target_class].sum()

    #                 # CHANGE 3: Add allow_unused=True to debug, but specifically 
    #                 # ensure the score is linked to the interpolated input.
    #                 grads = torch.autograd.grad(
    #                     outputs=score, 
    #                     inputs=interpolated,
    #                     retain_graph=False,
    #                     create_graph=False,
    #                     allow_unused=True # This prevents the crash
    #                 )[0]
                    
    #                 # If grads is None, it means the graph is still broken
    #                 if grads is None:
    #                     grads = torch.zeros_like(interpolated)
                
    #             integrated_grads += grads



    #         # Average and scale
    #         integrated_grads = integrated_grads / n_steps
    #         attributions = (waveform - baseline) * integrated_grads
            
    #         return attributions.detach()





# ===========================================================================================================================



    def compute_integrated_gradients(self, waveform, n_steps=50, target_class=1):
            self.model.eval()
            self.feature_extractor.eval()

            # 1. Get the baseline and input features from the extractor
            # We compute gradients relative to these features to avoid the "frontend block"
            with torch.no_grad():
                input_features_full = self.feature_extractor(waveform)
                # Use the same logic your model uses to pick hidden states
                input_features = input_features_full['hidden_states'][-1] if isinstance(input_features_full, dict) else input_features_full
                
            baseline_features = torch.zeros_like(input_features)
            alphas = torch.linspace(0, 1, n_steps, device=waveform.device)
            total_grads = torch.zeros_like(input_features)

            # 2. Integrate gradients over the backend model only
            for alpha in alphas:
                interpolated = baseline_features + alpha * (input_features - baseline_features)
                interpolated.requires_grad_(True)
                
                with torch.set_grad_enabled(True):
                    lengths = torch.full((interpolated.size(0),), interpolated.size(1), 
                                        dtype=torch.int16, device=waveform.device)
                    output = self.model(interpolated, lengths, dropout_prob=0.0)
                    
                    score = output.sum() if output.size(1) == 1 else output[:, target_class].sum()
                    grads = torch.autograd.grad(score, interpolated)[0]
                    
                total_grads += grads

            # 3. Compute Attribution at Feature Level
            feature_attribution = (input_features - baseline_features) * (total_grads / n_steps)
            
            # 4. Project back to time axis for the figure
            # Sum across the feature dimension (hidden_dim) to get a 1D time-importance map
            time_importance = feature_attribution.abs().sum(dim=-1).squeeze()
            
            # Interpolate/Stretch time_importance to match waveform length for visualization
            from torch.nn.functional import interpolate
            time_importance = interpolate(time_importance.view(1, 1, -1), 
                                        size=waveform.size(-1), 
                                        mode='linear').squeeze()
            
            return time_importance.detach()






    def visualize_attributions(
        self,
        waveform: np.ndarray,
        attributions: np.ndarray,
        sample_rate: int = 16000,
        segment_boundaries: Optional[List[Tuple[float, float]]] = None,
        save_path: str = 'outputs/integrated_gradients.png',
        title: str = 'Integrated Gradients Attribution'
    ):
        """
        Visualize integrated gradients attributions.
        
        Args:
            waveform: Original waveform (time,)
            attributions: Attribution values (time,)
            sample_rate: Audio sample rate
            segment_boundaries: Segment boundaries for PF audio
            save_path: Path to save figure
            title: Figure title
        """
        import librosa
        import librosa.display
        
        # Compute spectrogram
        S = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # 1. Waveform
        time_axis = np.linspace(0, len(waveform) / sample_rate, len(waveform))
        axes[0].plot(time_axis, waveform, linewidth=0.5, color='blue')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Original Waveform')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Spectrogram
        img = librosa.display.specshow(S_db, sr=sample_rate, x_axis='time', 
                                       y_axis='mel', ax=axes[1], cmap='viridis')
        axes[1].set_title('Mel Spectrogram')
        fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
        
        # 3. Attribution heatmap
        # Normalize attributions
        attr_norm = np.abs(attributions)
        attr_norm = (attr_norm - attr_norm.min()) / (attr_norm.max() - attr_norm.min() + 1e-8)
        
        axes[2].plot(time_axis, attr_norm, linewidth=1.5, color='red')
        axes[2].fill_between(time_axis, attr_norm, alpha=0.4, color='red')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Attribution')
        axes[2].set_title('Integrated Gradients Attribution (Importance)')
        axes[2].grid(True, alpha=0.3)
        
        # Mark segment boundaries
        if segment_boundaries:
            for start, end in segment_boundaries:
                for ax in axes:
                    ax.axvspan(start, end, alpha=0.2, color='yellow', label='Fake Segment')
            axes[0].legend(loc='upper right')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Integrated gradients visualization saved to: {save_path}")


def analyze_boundary_focus(
    attributions: np.ndarray,
    segment_boundaries: List[Tuple[int, int]],
    window_size: int = 1000
) -> Dict[str, float]:
    """
    Analyze whether gradients focus on segment boundaries.
    
    Args:
        attributions: Attribution values (time,)
        segment_boundaries: List of (start_sample, end_sample) indices
        window_size: Window size around boundaries (in samples)
    
    Returns:
        Dictionary with boundary focus statistics
    """
    # Get absolute attributions
    attr_abs = np.abs(attributions)
    
    # Identify boundary regions
    boundary_mask = np.zeros_like(attr_abs, dtype=bool)
    for start, end in segment_boundaries:
        # Mark regions around boundaries
        start_window = slice(max(0, start - window_size), 
                           min(len(attr_abs), start + window_size))
        end_window = slice(max(0, end - window_size), 
                         min(len(attr_abs), end + window_size))
        boundary_mask[start_window] = True
        boundary_mask[end_window] = True
    
    # Calculate statistics
    boundary_attr = attr_abs[boundary_mask]
    non_boundary_attr = attr_abs[~boundary_mask]
    
    return {
        'boundary_mean': float(np.mean(boundary_attr)) if len(boundary_attr) > 0 else 0,
        'non_boundary_mean': float(np.mean(non_boundary_attr)) if len(non_boundary_attr) > 0 else 0,
        'boundary_std': float(np.std(boundary_attr)) if len(boundary_attr) > 0 else 0,
        'focus_ratio': (float(np.mean(boundary_attr) / np.mean(non_boundary_attr)) 
                       if len(non_boundary_attr) > 0 and len(boundary_attr) > 0 else 0),
        'boundary_coverage': float(np.sum(boundary_mask) / len(boundary_mask))
    }


def compute_saliency_map(
    model: nn.Module,
    feature_extractor: nn.Module,
    waveform: torch.Tensor,
    target_class: int = 1
) -> torch.Tensor:
    """
    Compute simple saliency map (gradient w.r.t input).
    
    Faster than integrated gradients but less accurate.
    
    Args:
        model: Backend classification model
        feature_extractor: Feature extraction model
        waveform: Input waveform (1, time)
        target_class: Target class
    
    Returns:
        Saliency map (same shape as waveform)
    """
    waveform.requires_grad_(True)
    
    # Forward pass
    features_output = feature_extractor(waveform)
    if isinstance(features_output, dict):
        features = features_output['hidden_states'][-1]
    else:
        features = features_output
    
    # lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16, 
    #                     device=waveform.device)
    # output = model(features, lengths, dropout_prob=0.0)
    
    # # Backward pass
    # output[:, target_class].backward()


    lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16, 
                        device=waveform.device)
    output = model(features, lengths, dropout_prob=0.0)

    # Backward pass (handle single output dimension)
    if output.dim() == 1 or output.size(1) == 1:
        # Single output (binary classification with single neuron)
        output.sum().backward()
    else:
        # Multi-output (handle target_class)
        output[:, target_class].backward()

    
    saliency = waveform.grad.abs()
    
    # Clean up
    model.zero_grad()
    feature_extractor.zero_grad()
    
    return saliency.detach()


def compare_genuine_vs_pf_focus(
    ig_tool: IntegratedGradients,
    genuine_waveform: torch.Tensor,
    pf_waveform: torch.Tensor,
    pf_boundaries: List[Tuple[int, int]],
    save_path: str = 'outputs/focus_comparison.png'
):
    """
    Compare gradient focus between genuine and PF samples.
    
    Args:
        ig_tool: IntegratedGradients instance
        genuine_waveform: Genuine audio waveform
        pf_waveform: Partial-fake audio waveform
        pf_boundaries: Boundaries in PF audio
        save_path: Path to save comparison
    """
    # Compute attributions
    genuine_attr = ig_tool.compute_integrated_gradients(genuine_waveform).squeeze().cpu().numpy()
    pf_attr = ig_tool.compute_integrated_gradients(pf_waveform).squeeze().cpu().numpy()
    
    # Analyze PF focus on boundaries
    pf_stats = analyze_boundary_focus(pf_attr, pf_boundaries)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Genuine
    time_genuine = np.arange(len(genuine_attr))
    attr_genuine_norm = np.abs(genuine_attr)
    attr_genuine_norm = attr_genuine_norm / (attr_genuine_norm.max() + 1e-8)
    
    axes[0].plot(time_genuine, attr_genuine_norm, linewidth=1, color='green')
    axes[0].fill_between(time_genuine, attr_genuine_norm, alpha=0.3, color='green')
    axes[0].set_title('Genuine Audio - Attribution Pattern')
    axes[0].set_ylabel('Normalized Attribution')
    axes[0].grid(True, alpha=0.3)
    
    # Partial-Fake
    time_pf = np.arange(len(pf_attr))
    attr_pf_norm = np.abs(pf_attr)
    attr_pf_norm = attr_pf_norm / (attr_pf_norm.max() + 1e-8)
    
    axes[1].plot(time_pf, attr_pf_norm, linewidth=1, color='orange')
    axes[1].fill_between(time_pf, attr_pf_norm, alpha=0.3, color='orange')
    
    # Mark boundaries
    for start, end in pf_boundaries:
        axes[1].axvspan(start, end, alpha=0.2, color='red', label='Fake Segment')
    
    axes[1].set_title(f'Partial-Fake Audio - Attribution Pattern '
                     f'(Boundary Focus Ratio: {pf_stats["focus_ratio"]:.2f})')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Normalized Attribution')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.suptitle('Gradient Focus Comparison: Genuine vs Partial-Fake', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Focus comparison saved to: {save_path}")
    print(f"\nPF Boundary Focus Statistics:")
    print(f"  Boundary mean: {pf_stats['boundary_mean']:.4f}")
    print(f"  Non-boundary mean: {pf_stats['non_boundary_mean']:.4f}")
    print(f"  Focus ratio: {pf_stats['focus_ratio']:.2f}")


# ============================================================================
# Example Usage
# ============================================================================

# if __name__ == "__main__":
#     print("Gradient-based Analysis Example")
#     print("=" * 60)
#     print("\nThis module provides:")
#     print("1. Integrated Gradients - find important input regions")
#     print("2. Boundary Focus Analysis - check if model focuses on PF boundaries")
#     print("3. Saliency Maps - fast gradient visualization")
#     print("\nExample usage:")
#     print("""
#     from utils.gradient_analysis import IntegratedGradients
    
#     ig = IntegratedGradients(model, feature_extractor)
#     attributions = ig.compute_integrated_gradients(waveform)
#     ig.visualize_attributions(
#         waveform.cpu().numpy(),
#         attributions.cpu().numpy(),
#         segment_boundaries=[(0.5, 1.2), (2.0, 2.8)]
#     )
#     """)



if __name__ == "__main__":
    """
    Standalone script to run gradient analysis on inference results.
    
    Usage:
        python -m utils.gradient_analysis --config config/default_config.yaml \
            --audio-file path/to/audio.wav \
            --boundaries 0.5 1.2 2.0 2.8 \
            --n-steps 50
    """
    import argparse
    import torchaudio
    from utils.config_manager import ConfigManager
    from model import initialize_models
    
    parser = argparse.ArgumentParser(description='Gradient Analysis on Audio File')
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                       help='Path to configuration file')
    # parser.add_argument('--audio-file', type=str, required=True,
    #                    help='Path to audio file for analysis')
    parser.add_argument('--boundaries', type=float, nargs='*', default=None,
                       help='Segment boundaries in seconds (e.g., 0.5 1.2 2.0 2.8)')
    parser.add_argument('--n-steps', type=int, default=50,
                       help='Number of integration steps for Integrated Gradients')
    # parser.add_argument('--output-dir', type=str, default='outputs/gradient_analysis',
    #                    help='Output directory for visualizations')
    # parser.add_argument('--output-dir', type=str, default='outputs/gradient_analysis2',
    #                    help='Output directory for visualizations')
    parser.add_argument('--output-dir', type=str, default='outputs/gradient_analysis3',
                       help='Output directory for visualizations')
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigManager(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("INTEGRATED GRADIENTS ANALYSIS")
    print("="*80)
    
    # Check if audio file exists
    # if not os.path.exists(args.audio_file):
    #     print(f"⚠️  Audio file not found: {args.audio_file}")
    #     exit(1)
    
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
    



    import json
    file_path="/root/Partial_Spoof_Detection_System/outputs/reports/hardest_samples_20260117_094825.json"
    with open(file_path, "r") as f:
        data = json.load(f)

    genuine_file_names = [item["file"] for item in data["genuine"]]
    spoof_file_names = [item["file"] for item in data["spoof"]]


    # random_file_names
    directory = "/root/Partial_Spoof_Detection_System/database/RFP_2/testing_subset"
    random_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(directory)
        if f.lower().endswith(".wav")
    ]
    
    hard_correct_prediction_files = genuine_file_names + spoof_file_names + random_file_names
    # hard_correct_prediction_files = genuine_file_names + spoof_file_names 
    for file_name in hard_correct_prediction_files:
        print(file_name)


        # Load audio
        print(f"\n2. Loading audio: {file_name}")
        waveform, sr = torchaudio.load(f"/root/Partial_Spoof_Detection_System/database/RFP/testing/{file_name}.wav")
        waveform = waveform.to(device)
        
        # Compute integrated gradients
        print(f"\n3. Computing Integrated Gradients ({args.n_steps} steps)...")
        ig = IntegratedGradients(model, feature_extractor)
        attributions = ig.compute_integrated_gradients(waveform, n_steps=args.n_steps)
        
        # Visualize attributions
        print("\n4. Generating visualization...")
        os.makedirs(args.output_dir, exist_ok=True)
        
        ig.visualize_attributions(
            waveform.squeeze().cpu().numpy(),
            attributions.squeeze().cpu().numpy(),
            sample_rate=sr,
            segment_boundaries=segment_boundaries,
            save_path=os.path.join(args.output_dir, f'integrated_gradients_{file_name}.png'),
            title=f'Integrated Gradients: {file_name}'
        )
        
        # Analyze boundary focus if boundaries provided
        if segment_boundaries:
            print("\n5. Analyzing boundary focus...")
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
            
            print(f"\nBoundary Focus Statistics:")
            print(f"  Boundary mean: {stats['boundary_mean']:.4f}")
            print(f"  Non-boundary mean: {stats['non_boundary_mean']:.4f}")
            print(f"  Focus ratio: {stats['focus_ratio']:.2f}")
            print(f"  Boundary coverage: {stats['boundary_coverage']:.2%}")
            
            # Save statistics to JSON
            import json
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            stats_path = os.path.join(args.output_dir, f'boundary_focus_{timestamp}.json')
            with open(stats_path, 'w') as f:
                json.dump({
                    'file': args.audio_file,
                    'boundaries': segment_boundaries,
                    'n_steps': args.n_steps,
                    'statistics': stats
                }, f, indent=2)
            print(f"\n✓ Statistics saved to: {stats_path}")
        
        print(f"\n✓ Gradient analysis complete!")
        print(f"  Results saved to: {args.output_dir}")

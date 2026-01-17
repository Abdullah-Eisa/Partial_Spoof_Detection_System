
"""
Batch Comprehensive Evaluation
Process multiple audio files for attention/gradient analysis

Usage:
    python batch_comprehensive_evaluation.py \
        --config config/default_config.yaml \
        --audio-dir database/partial_fake_samples/ \
        --boundaries-file boundaries.json
"""

import os
import argparse
import json
import glob
from datetime import datetime
from tqdm import tqdm

from utils.config_manager import ConfigManager
from utils.attention_visualization import AttentionExtractor, visualize_attention_on_spectrogram
from utils.gradient_analysis import IntegratedGradients, analyze_boundary_focus

import torch
from model import initialize_models


def load_boundaries_file(boundaries_path):
    """
    Load segment boundaries from JSON file.
    
    Expected format:
    {
        "sample1.wav": [[0.5, 1.2], [2.0, 2.8]],
        "sample2.wav": [[1.0, 1.5]],
        ...
    }
    """
    if not os.path.exists(boundaries_path):
        return {}
    
    with open(boundaries_path, 'r') as f:
        boundaries = json.load(f)
    
    return boundaries


def process_audio_directory(
    audio_dir,
    boundaries_dict,
    model,
    feature_extractor,
    output_dir,
    device='cpu',
    analyze_attention=True,
    analyze_gradients=True
):
    """Process all audio files in directory."""
    
    # Find all .wav files
    audio_files = glob.glob(os.path.join(audio_dir, '*.wav'))
    
    if not audio_files:
        print(f"⚠️  No .wav files found in {audio_dir}")
        return
    
    print(f"\nFound {len(audio_files)} audio files")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        filename = os.path.basename(audio_path)
        file_output_dir = os.path.join(output_dir, filename.replace('.wav', ''))
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Get boundaries for this file
        boundaries = boundaries_dict.get(filename, None)
        
        # Load audio
        import torchaudio
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.to(device)
        
        try:
            # Attention analysis
            if analyze_attention:
                print(f"\n  Analyzing attention: {filename}")
                extractor = AttentionExtractor(model)
                attention_dict = extractor.extract_attention(waveform, feature_extractor)
                
                if 'time_pooling' in attention_dict:
                    attention = attention_dict['time_pooling'].squeeze().numpy()
                    attention = (attention - attention.min()) / (attention.max() - attention.min())
                    
                    visualize_attention_on_spectrogram(
                        waveform.squeeze().cpu().numpy(),
                        attention,
                        sample_rate=sr,
                        segment_boundaries=boundaries,
                        save_path=os.path.join(file_output_dir, 'attention.png'),
                        title=f'Attention: {filename}'
                    )
                
                extractor.remove_hooks()
            
            # Gradient analysis
            if analyze_gradients:
                print(f"  Computing gradients: {filename}")
                ig = IntegratedGradients(model, feature_extractor)
                attributions = ig.compute_integrated_gradients(waveform, n_steps=50)
                
                ig.visualize_attributions(
                    waveform.squeeze().cpu().numpy(),
                    attributions.squeeze().cpu().numpy(),
                    sample_rate=sr,
                    segment_boundaries=boundaries,
                    save_path=os.path.join(file_output_dir, 'gradients.png'),
                    title=f'Integrated Gradients: {filename}'
                )
                
                # Analyze boundary focus if boundaries provided
                if boundaries:
                    frame_boundaries = [
                        (int(start * sr), int(end * sr))
                        for start, end in boundaries
                    ]
                    
                    stats = analyze_boundary_focus(
                        attributions.squeeze().cpu().numpy(),
                        frame_boundaries,
                        window_size=int(0.1 * sr)
                    )
                    
                    # Save statistics
                    with open(os.path.join(file_output_dir, 'boundary_stats.json'), 'w') as f:
                        json.dump({
                            'file': filename,
                            'boundaries': boundaries,
                            'stats': stats
                        }, f, indent=2)
        
        except Exception as e:
            print(f"  ⚠️  Error processing {filename}: {str(e)}")
            continue
    
    print(f"\n✓ Batch processing complete!")
    print(f"  Results saved to: {output_dir}")


def main(args):
    """Main batch evaluation."""
    print("\n" + "="*80)
    print("BATCH COMPREHENSIVE EVALUATION")
    print("="*80)
    
    # Load configuration
    config = ConfigManager(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("\nLoading model...")
    model, feature_extractor, _ = initialize_models(
        config, save_feature_extractor=False, LEARNING_RATE=0.0001, DEVICE=device
    )
    
    checkpoint = torch.load(config['paths']['ps_model_checkpoint'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    feature_extractor.eval()
    
    # Load boundaries
    boundaries_dict = {}
    if args.boundaries_file:
        print(f"Loading boundaries from: {args.boundaries_file}")
        boundaries_dict = load_boundaries_file(args.boundaries_file)
        print(f"  Loaded boundaries for {len(boundaries_dict)} files")
    else:
        print("⚠️  No boundaries file provided - will analyze without boundaries")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'outputs/batch_evaluation_{timestamp}'
    
    # Process directory
    process_audio_directory(
        args.audio_dir,
        boundaries_dict,
        model,
        feature_extractor,
        output_dir,
        device=device,
        analyze_attention=not args.skip_attention,
        analyze_gradients=not args.skip_gradients
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch Comprehensive Evaluation')
    
    parser.add_argument('--config', type=str, default='config/default_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--audio-dir', type=str, required=True,
                      help='Directory containing .wav files')
    parser.add_argument('--boundaries-file', type=str, default=None,
                      help='JSON file with segment boundaries')
    
    parser.add_argument('--skip-attention', action='store_true',
                      help='Skip attention analysis')
    parser.add_argument('--skip-gradients', action='store_true',
                      help='Skip gradient analysis')
    
    args = parser.parse_args()
    main(args)

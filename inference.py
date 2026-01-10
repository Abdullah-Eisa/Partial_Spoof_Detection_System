
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from datetime import datetime

from utils import *
from preprocess import *
from model import *

import os
from tqdm import tqdm
import torch
import torch.nn as nn
from datetime import datetime
from utils.config_manager import ConfigManager
from utils.utils import *
from preprocess import *
from model import *

from utils.parameter_counter import (
    count_parameters, 
    print_inference_model_info,
    quick_param_count,
    get_model_size_mb,
    get_block_parameter_breakdown,
    print_block_parameter_breakdown,
    print_comprehensive_model_analysis
)


def inference_helper(model, feature_extractor, criterion,
                  test_data_loader, DEVICE='cpu'):
    """Evaluate the model on the test set and compute Precision, Recall, F1 metrics"""

    # testing phase
    model.eval()  # Set the model to evaluation mode

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(DEVICE)
        print("Parallelizing model on ", torch.cuda.device_count(), "GPUs!")

    # Initialize variables
    files_names = []

    epoch_loss = 0
    utterance_predictions = []
    utterance_labels = []
    dropout_prob = 0
    nan_count = 0 # To count the number of NaNs in the loss
    
    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc="Test Batches", leave=False):
            waveforms = batch['waveform'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            labels = labels.unsqueeze(1).float()   # Converts labels from shape [batch_size] to [batch_size, 1]

            # Forward pass through feature extractor
            features_output = feature_extractor(waveforms)

            # Handle both dictionary output (SSL models) and direct tensor output (MFCC/LFCC)
            if isinstance(features_output, dict):
                features = features_output['hidden_states'][-1]
            else:
                features = features_output

            # lengths should be the number of non-padded frames in each sequence
            lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16).to(DEVICE)  # (batch_size,)

            # Pass features to model and get predictions
            outputs = forward_pass(model, features, lengths, dropout_prob)

            # Calculate loss
            loss = criterion(outputs, labels) 
            if torch.isnan(loss).any(): 
                print(f"NaN detected in test loop loss") 
                nan_count += torch.isnan(loss).sum().item()
                print(f"loss value: {loss.item()}")
                print(f"batch['file_name']: {batch['file_name']}")
                print(f"in inference_helper batch, nan_count: {nan_count}")
                continue

            epoch_loss += loss.item()

            with torch.no_grad():
                utterance_predictions.extend(outputs)
                utterance_labels.extend(labels)
                files_names.extend(batch['file_name'])

        # Get Average Utterance EER for the epoch
        utterance_labels = torch.cat(utterance_labels)
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_metrics(utterance_predictions, utterance_labels)
        
        # Compute Precision, Recall, and F1
        precision, recall, f1 = compute_precision_recall_f1(utterance_predictions, utterance_labels)

        # Average loss for the epoch
        epoch_loss /= len(test_data_loader)


    # Print epoch testing results
    print(f'Testing/Inference Complete. Test Loss: {epoch_loss:.4f},\n'
               f'Average Test Utterance EER: {utterance_eer:.4f}, Average Test Utterance EER Threshold: {utterance_eer_threshold:.4f}')
    print("===================================================")
    print(f'In Test loop, Total loss NAN count: {nan_count}')

    return create_metrics_dict(utterance_eer, utterance_eer_threshold, epoch_loss, precision, recall, f1)




# def inference(config):
#     """Run inference using configuration"""
#     print("Starting inference...")
    
#     device = torch.device(config['system']['device'])
#     print(f"Using device: {device}")

#     # Initialize data loader
#     eval_data_loader = initialize_data_loader(
#         dataset_name=config['data']['dataset_name'],
#         data_path=config['data']['eval_data_path'],
#         labels_path=config['data']['eval_labels_path'],
#         BATCH_SIZE=config['inference'].get('batch_size', config['training']['batch_size']),
#         shuffle=False,
#         num_workers=config['inference'].get('num_workers', config['system']['num_workers']),
#         prefetch_factor=config['inference'].get('prefetch_factor', config['system']['prefetch_factor']),
#         pin_memory=config['inference'].get('pin_memory', config['system']['pin_memory'])
#     )

#     # Load models
#     feature_extractor = torch.hub.load('s3prl/s3prl', 'wav2vec2', 
#                                      model_path=config['paths']['ssl_checkpoint']).to(device)
#     feature_extractor.eval()

#     PS_Model = BinarySpoofingClassificationModel(
#         feature_dim=config['model']['feature_dim'],
#         num_heads=config['model']['num_heads'],
#         hidden_dim=config['model']['hidden_dim'],
#         max_dropout=config['model']['max_dropout'],
#         depthwise_conv_kernel_size=config['model']['depthwise_conv_kernel_size'],
#         conformer_layers=config['model']['conformer_layers'],
#         max_pooling_factor=config['model']['max_pooling_factor']
#     ).to(device)

#     # Load model checkpoint
#     try:
#         checkpoint = torch.load(config['paths']['ps_model_checkpoint'], map_location=device)
#         PS_Model.load_state_dict(checkpoint['model_state_dict'])
#         print(f"Loaded model checkpoint from {config['paths']['ps_model_checkpoint']}")
#     except Exception as e:
#         print(f"Error loading model checkpoint: {str(e)}")
#         return

#     PS_Model.eval()

#     criterion = initialize_loss_function().to(device)
    
#     # Call inference helper function
#     results = inference_helper(
#         model=PS_Model,
#         feature_extractor=feature_extractor,
#         criterion=criterion,
#         test_data_loader=eval_data_loader, 
#         DEVICE=device
#     )

#     if device == 'cuda':
#         torch.cuda.empty_cache()
        
#     return results


# def inference(config):
#     """Run inference using configuration"""
#     from feature_extractors import FeatureExtractorFactory
    
#     print("Starting inference...")
    
#     device = torch.device(config['system']['device'])
    
#     # Initialize data loader
#     eval_data_loader = initialize_data_loader(
#         dataset_name=config['data']['dataset_name'],
#         data_path=config['data']['eval_data_path'],
#         labels_path=config['data']['eval_labels_path'],
#         BATCH_SIZE=config['inference'].get('batch_size', config['training']['batch_size']),
#         shuffle=False,
#         num_workers=config['inference'].get('num_workers', config['system']['num_workers']),
#         prefetch_factor=config['inference'].get('prefetch_factor', config['system']['prefetch_factor']),
#         pin_memory=config['inference'].get('pin_memory', config['system']['pin_memory'])
#     )

#     # Load feature extractor
#     feature_extractor = FeatureExtractorFactory.create_extractor(config, device)
#     feature_extractor.eval()

#     # Get feature dimension
#     from feature_extractors import get_feature_dim_from_config
#     feature_dim = get_feature_dim_from_config(config)

#     PS_Model = BinarySpoofingClassificationModel(
#         feature_dim=feature_dim,
#         num_heads=config['model']['num_heads'],
#         hidden_dim=config['model']['hidden_dim'],
#         max_dropout=config['model']['max_dropout'],
#         depthwise_conv_kernel_size=config['model']['depthwise_conv_kernel_size'],
#         conformer_layers=config['model']['conformer_layers'],
#         max_pooling_factor=config['model'].get('max_pooling_factor'),
#         use_max_pooling=config['model'].get('use_max_pooling', True),
#         pooling_strategy=config['model'].get('pooling_strategy', 'self_weighted'),
#         config=config
#     ).to(device)

#     # Load model checkpoint
#     try:
#         checkpoint = torch.load(config['paths']['ps_model_checkpoint'], map_location=device)
#         PS_Model.load_state_dict(checkpoint['model_state_dict'])
#         print(f"Loaded model checkpoint from {config['paths']['ps_model_checkpoint']}")
#     except Exception as e:
#         print(f"Error loading model checkpoint: {str(e)}")
#         return

#     PS_Model.eval()

#     criterion = initialize_loss_function().to(device)
    
#     # Call inference helper function
#     results = inference_helper(
#         model=PS_Model,
#         feature_extractor=feature_extractor,
#         criterion=criterion,
#         test_data_loader=eval_data_loader, 
#         DEVICE=device
#     )

#     if device == 'cuda':
#         torch.cuda.empty_cache()
        
#     return results


def inference(config, show_model_info: bool = True, save_model_info: bool = False):
    """Run inference using configuration with sequence model support"""
    from feature_extractors import FeatureExtractorFactory
    
    print("Starting inference...")
    
    device = torch.device(config['system']['device'])
    
    # Initialize data loader
    eval_data_loader = initialize_data_loader(
        dataset_name=config['data']['dataset_name'],
        data_path=config['data']['eval_data_path'],
        labels_path=config['data']['eval_labels_path'],
        BATCH_SIZE=config['inference'].get('batch_size', config['training']['batch_size']),
        shuffle=False,
        num_workers=config['inference'].get('num_workers', config['system']['num_workers']),
        prefetch_factor=config['inference'].get('prefetch_factor', config['system']['prefetch_factor']),
        pin_memory=config['inference'].get('pin_memory', config['system']['pin_memory'])
    )

    # Load feature extractor
    finetuned_checkpoint = config['feature_extractor'].get('finetuned_checkpoint', None)
    if finetuned_checkpoint:
        print(f"Loading feature extractor with finetuned weights from config: {finetuned_checkpoint}")
    
    feature_extractor = FeatureExtractorFactory.create_extractor(config, device)
    feature_extractor.eval()

    # Get feature dimension
    from feature_extractors import get_feature_dim_from_config
    feature_dim = get_feature_dim_from_config(config)
    
    # Get sequence model configuration
    sequence_model_type = config['model'].get('sequence_model_type', 'conformer')
    sequence_model_config = config['model'].get('sequence_model_config', None)

    print(f"Loading model with sequence type: {sequence_model_type}")

    PS_Model = BinarySpoofingClassificationModel(
        feature_dim=feature_dim,
        num_heads=config['model']['num_heads'],
        hidden_dim=config['model']['hidden_dim'],
        max_dropout=config['model']['max_dropout'],
        depthwise_conv_kernel_size=config['model']['depthwise_conv_kernel_size'],
        conformer_layers=config['model']['conformer_layers'],
        max_pooling_factor=config['model'].get('max_pooling_factor'),
        use_max_pooling=config['model'].get('use_max_pooling', True),
        pooling_strategy=config['model'].get('pooling_strategy', 'self_weighted'),
        sequence_model_type=sequence_model_type,
        sequence_model_config=sequence_model_config,
        config=config
    ).to(device)

    # Load model checkpoint
    try:
        checkpoint = torch.load(config['paths']['ps_model_checkpoint'], map_location=device)
        PS_Model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model checkpoint from {config['paths']['ps_model_checkpoint']}")
    except Exception as e:
        print(f"Error loading model checkpoint: {str(e)}")
        return

    PS_Model.eval()


    # ====== NEW: Display model parameter information ======
    if show_model_info:
        # Print comprehensive model information with block breakdown
        model_stats = print_inference_model_info(
            model=PS_Model,
            feature_extractor=feature_extractor,
            show_breakdown=True
        )
        
        # Display block-wise parameter breakdown for backend model
        print("\n" + "="*120)
        print("BACKEND MODEL - BLOCK-WISE PARAMETER ANALYSIS")
        print("="*120)
        backend_blocks = get_block_parameter_breakdown(
            PS_Model,
            block_patterns={
                'Downsampling': ['downsample', 'pooling_layer', 'feature_projection'],
                'Sequence_Model': ['sequence_model', 'conformer', 'lstm', 'transformer', 'cnn', 'tcn'],
                'Pooling': ['pooling', 'sap', 'self_weighted'],
                'FC_Refinement': ['fc_refinement', 'classification']
            },
            verbose=True
        )
        
        # Display comprehensive analysis if feature extractor is provided
        if feature_extractor is not None:
            print_comprehensive_model_analysis(
                model=PS_Model,
                feature_extractor=feature_extractor,
                config=config
            )
        
        # Save to file if requested
        if save_model_info:
            save_model_info_to_file(
                model=PS_Model,
                feature_extractor=feature_extractor,
                config=config,
                stats=model_stats,
                backend_blocks=backend_blocks,
                output_dir='outputs'
            )
    else:
        # Just print a quick summary
        print("\nModel Summary:")
        backend_total, backend_train, _ = quick_param_count(PS_Model)
        fe_total, fe_train, _ = quick_param_count(feature_extractor)
        print(f"  Backend Model:      {backend_train:>12,} trainable params")
        print(f"  Feature Extractor:  {fe_train:>12,} trainable params")
        print(f"  Total System:       {backend_train + fe_train:>12,} trainable params")
        print(f"  Memory (float32):   {get_model_size_mb(PS_Model) + get_model_size_mb(feature_extractor):>11.2f} MB\n")
    # ====== END NEW ======


    criterion = initialize_loss_function().to(device)
    
    # Call inference helper function
    results = inference_helper(
        model=PS_Model,
        feature_extractor=feature_extractor,
        criterion=criterion,
        test_data_loader=eval_data_loader, 
        DEVICE=device
    )

    if device == 'cuda':
        torch.cuda.empty_cache()
        
    return results



def save_model_info_to_file(model, feature_extractor, config, stats, backend_blocks=None, output_dir='outputs'):
    """
    Save detailed model information to a text file.
    
    Args:
        model: Backend classification model
        feature_extractor: Feature extraction model
        config: Configuration dictionary
        stats: Statistics dictionary from print_inference_model_info
        backend_blocks: Block-wise parameter breakdown from get_block_parameter_breakdown
        output_dir: Directory to save the output file
    """
    import sys
    from io import StringIO
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_name = config['data']['dataset_name']
    sequence_type = config['model'].get('sequence_model_type', 'conformer')
    filename = f"inference_model_info_{dataset_name}_{sequence_type}_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Redirect stdout to capture print statements
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    # Print all information
    print("="*80)
    print("INFERENCE MODEL INFORMATION")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dataset: {dataset_name}")
    print(f"Sequence Model: {sequence_type}")
    print(f"Checkpoint: {config['paths']['ps_model_checkpoint']}")
    print(f"Device: {config['system']['device']}")
    
    # Print model architecture details
    print("\n" + "="*80)
    print_inference_model_info(model, feature_extractor, show_breakdown=True)
    
    # Print block-wise breakdown if provided
    if backend_blocks:
        print("\n" + "="*80)
        print("BACKEND MODEL - BLOCK-WISE PARAMETER BREAKDOWN")
        print("="*80)
        print_block_parameter_breakdown(backend_blocks)
    
    # Print configuration summary
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Feature Extractor Type:    {config['feature_extractor']['type']}")
    print(f"Pooling Strategy:          {config['model'].get('pooling_strategy', 'self_weighted')}")
    print(f"Number of Heads:           {config['model']['num_heads']}")
    print(f"Hidden Dimension:          {config['model']['hidden_dim']}")
    print(f"Conformer Layers:          {config['model']['conformer_layers']}")
    print(f"Max Dropout:               {config['model']['max_dropout']}")
    print(f"Batch Size (Inference):    {config['inference'].get('batch_size', 'N/A')}")
    print("="*80 + "\n")

    
    # Get captured output
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    # Save to file
    with open(filepath, 'w') as f:
        f.write(output)
    
    print(f"\n✓ Model information saved to: {filepath}")
    print(f"  File size: {os.path.getsize(filepath) / 1024:.2f} KB")


def dev_one_epoch(model, feature_extractor, criterion,
                  dev_data_loader, dropout_prob=0, DEVICE='cpu'):
    """Evaluate the model on the development set and compute Precision, Recall, F1 metrics"""

    # Validation phase
    model.eval()  # Set the model to evaluation mode

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(DEVICE)
        print("Parallelizing model on ", torch.cuda.device_count(), "GPUs!")

    # Initialize variables
    files_names = []

    epoch_loss = 0
    utterance_eer, utterance_eer_threshold = 0, 0
    utterance_predictions = []
    utterance_labels = []
    nan_count = 0 # To count the number of NaNs in the loss
    
    with torch.no_grad():
        for batch in tqdm(dev_data_loader, desc="Dev Batches", leave=False):
            waveforms = batch['waveform'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            labels = labels.unsqueeze(1).float()   # Converts labels from shape [batch_size] to [batch_size, 1]

            # Forward pass through feature extractor
            features_output = feature_extractor(waveforms)

            # Handle both dictionary output (SSL models) and direct tensor output (MFCC/LFCC)
            if isinstance(features_output, dict):
                features = features_output['hidden_states'][-1]
            else:
                features = features_output

            # lengths should be the number of non-padded frames in each sequence
            lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16).to(DEVICE)  # (batch_size,)

            # Pass features to model and get predictions
            outputs = forward_pass(model, features, lengths, dropout_prob)

            # Calculate loss
            loss = criterion(outputs, labels) 
            if torch.isnan(loss).any(): 
                print(f"NaN detected in loss during development loop")
                nan_count += torch.isnan(loss).sum().item()
                print(f"loss value: {loss.item()}")
                print(f"batch['file_name']: {batch['file_name']}")
                print(f"in dev_one_epoch batch, nan_count: {nan_count}")
                continue

            epoch_loss += loss.item()

            with torch.no_grad():
                utterance_predictions.extend(outputs)
                utterance_labels.extend(labels)
                files_names.extend(batch['file_name'])

        # Get Average Utterance EER for the epoch
        utterance_labels = torch.cat(utterance_labels)
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_metrics(utterance_predictions, utterance_labels)
        
        # Compute Precision, Recall, and F1
        precision, recall, f1 = compute_precision_recall_f1(utterance_predictions, utterance_labels)

        # Average loss for the epoch
        epoch_loss /= len(dev_data_loader)

    # Print epoch dev progress
    print("===================================================")
    print(f'In Dev loop, Total loss NAN count: {nan_count}')
    
    return create_metrics_dict(utterance_eer, utterance_eer_threshold, epoch_loss, precision, recall, f1), nan_count



if __name__ == "__main__":

    config = ConfigManager()
    start_time = datetime.now()
    
    try:
        # Run inference with model info display and save to file
        results = inference(
            config, 
            show_model_info=True,  # Set to False to skip detailed display
            save_model_info=True   # Set to False to skip saving to file
        )
        
        if results:
            print("\n" + "="*80)
            print("INFERENCE RESULTS")
            print(results)
            print("="*80)
            print(f"Utterance EER:           {results['utterance_eer']:.4f}")
            print(f"Utterance EER Threshold: {results['utterance_eer_threshold']:.4f}")
            print(f"Epoch Loss:              {results['epoch_loss']:.4f}")
            print(f"Precision:               {results.get('precision', 'N/A'):.4f}" if 'precision' in results else f"Precision:               N/A")
            print(f"Recall:                  {results.get('recall', 'N/A'):.4f}" if 'recall' in results else f"Recall:                  N/A")
            print(f"F1 Score:                {results.get('f1', 'N/A'):.4f}" if 'f1' in results else f"F1 Score:                N/A")
            print("="*80 + "\n")
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()


    end_time = datetime.now()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Total time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")


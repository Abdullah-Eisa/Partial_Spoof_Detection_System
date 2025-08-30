
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

def inference_helper(model, feature_extractor,criterion,
                  test_data_loader, DEVICE='cpu'):
    """Evaluate the model on the test set"""

    # testing phase
    model.eval()  # Set the model to evaluation mode

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(DEVICE)
        print("Parallelizing model on ", torch.cuda.device_count(), "GPUs!")

    # Initialize variables
    files_names=[]

    epoch_loss = 0
    utterance_predictions=[]
    utterance_labels=[]
    dropout_prob=0
    nan_count=0 # To count the number of NaNs in the loss
    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc="Test Batches", leave=False):
            waveforms = batch['waveform'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            labels = labels.unsqueeze(1).float()   # Converts labels from shape [batch_size] to [batch_size, 1]

            # Forward pass through wav2vec2 for feature extraction
            features = feature_extractor(waveforms)['hidden_states'][-1] 

            # lengths should be the number of non-padded frames in each sequence
            lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16).to(DEVICE)  # (batch_size,)

            # Pass features to model and get predictions
            outputs = forward_pass(model, features, lengths, dropout_prob)

            # Calculate loss
            loss = criterion(outputs, labels) 
            if torch.isnan(loss).any(): 
                print(f"NaN detected in test loop loss") 
                nan_count+=torch.isnan(loss).sum().item()
                print(f"loss value: {loss.item()}")
                print(f"batch['file_name']: {batch['file_name']}")
                print(f"in train_one_epoch batch, nan_count: {nan_count}")
                continue

            epoch_loss += loss.item()

            with torch.no_grad():
                utterance_predictions.extend(outputs)
                utterance_labels.extend(labels)
                files_names.extend(batch['file_name'])


        # Get Average Utterance EER for the epoch
        utterance_labels = torch.cat(utterance_labels)
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_metrics(utterance_predictions,utterance_labels)

        # Average loss for the epoch
        epoch_loss /= len(test_data_loader)


    # Print epoch testing results
    print(f'Testing/Inference Complete. Test Loss: {epoch_loss:.4f},\n'
               f'Average Test Utterance EER: {utterance_eer:.4f}, Average Test Utterance EER Threshold: {utterance_eer_threshold:.4f}')
    print("===================================================")
    print(f'In Test loop, Total loss NAN count: {nan_count}')

    return create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_loss)




def inference(config):
    """Run inference using configuration"""
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

    # Load models
    feature_extractor = torch.hub.load('s3prl/s3prl', 'wav2vec2', 
                                     model_path=config['paths']['ssl_checkpoint']).to(device)
    feature_extractor.eval()

    PS_Model = BinarySpoofingClassificationModel(
        feature_dim=config['model']['feature_dim'],
        num_heads=config['model']['num_heads'],
        hidden_dim=config['model']['hidden_dim'],
        max_dropout=config['model']['max_dropout'],
        depthwise_conv_kernel_size=config['model']['depthwise_conv_kernel_size'],
        conformer_layers=config['model']['conformer_layers'],
        max_pooling_factor=config['model']['max_pooling_factor']
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




def dev_one_epoch(model, feature_extractor,criterion,
                  dev_data_loader,dropout_prob=0,DEVICE='cpu'):
    """Evaluate the model on the development set"""

    # Validation phase
    model.eval()  # Set the model to evaluation mode

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(DEVICE)
        print("Parallelizing model on ", torch.cuda.device_count(), "GPUs!")

    # Initialize variables
    files_names=[]

    epoch_loss = 0
    utterance_eer, utterance_eer_threshold=0,0
    utterance_predictions=[]
    utterance_labels=[]
    nan_count=0 # To count the number of NaNs in the loss
    with torch.no_grad():
        for batch in tqdm(dev_data_loader, desc="Dev Batches", leave=False):
            waveforms = batch['waveform'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            labels = labels.unsqueeze(1).float()   # Converts labels from shape [batch_size] to [batch_size, 1]

            # Forward pass through wav2vec2 for feature extraction
            features = feature_extractor(waveforms)['hidden_states'][-1] 

            # lengths should be the number of non-padded frames in each sequence
            lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16).to(DEVICE)  # (batch_size,)

            # Pass features to model and get predictions
            outputs = forward_pass(model, features, lengths, dropout_prob)

            # Calculate loss
            loss = criterion(outputs, labels) 
            if torch.isnan(loss).any(): 
                print(f"NaN detected in loss during development loop")
                # c+=1
                nan_count+=torch.isnan(loss).sum().item()
                print(f"loss value: {loss.item()}")
                print(f"batch['file_name']: {batch['file_name']}")
                print(f"in dev_one_epoch batch, nan_count: {nan_count}")
                continue

            epoch_loss += loss.item()

            with torch.no_grad():
                utterance_predictions.extend(outputs)                 # Calculate utterance predictions
                utterance_labels.extend(labels)
                files_names.extend(batch['file_name'])                 # Accumulate files names


        # Get Average Utterance EER for the epoch
        utterance_labels = torch.cat(utterance_labels)
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_metrics(utterance_predictions,utterance_labels)

        # Average loss for the epoch
        epoch_loss /= len(dev_data_loader)


    # Print epoch dev progress
    # print(f'Epoch [{epoch + 1}] Complete. Validation Loss: {epoch_loss:.4f},\n'
    #            f'Average Validation Segment EER: {segment_eer:.4f}, Average Validation Segment EER Threshold: {segment_eer_threshold:.4f},\n'
    #            f'Average Validation Utterance EER: {utterance_eer:.4f}, Average Validation Utterance EER Threshold: {utterance_eer_threshold:.4f}')
    print("===================================================")
    print(f'In Dev loop, Total loss NAN count: {nan_count}')
    
    return create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_loss), nan_count



if __name__ == "__main__":

    config = ConfigManager()
    start_time = datetime.now()
    
    try:
        results = inference(config)
        if results:
            print("Inference results:", results)
    except Exception as e:
        print(f"Error during inference: {str(e)}")
    
    end_time = datetime.now()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Total time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")


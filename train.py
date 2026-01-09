import os
import torch
import wandb
from torch.optim import lr_scheduler
from datetime import datetime
from tqdm import tqdm

import matplotlib.pyplot as plt

from utils.utils import *
from preprocess import *
from model import *
from inference import dev_one_epoch

# ===========================================================================================================================
# Define training logic for one epoch
def train_one_epoch(model, train_loader, feature_extractor, optimizer, criterion, max_grad_norm, dropout_prob=0, DEVICE='cpu'):
    """Train for one epoch and compute Precision, Recall, F1 metrics"""
    model.train()

    epoch_loss = 0
    utterance_predictions = []
    utterance_labels = []
    files_names = []
    nan_count = 0
    
    for batch in tqdm(train_loader, desc="Train Batches", leave=False):
        waveforms = batch['waveform'].to(DEVICE)
        labels = batch['label'].to(DEVICE).unsqueeze(1).float()

        optimizer.zero_grad()

        # Feature extraction
        features_output = feature_extractor(waveforms)

        # Handle both dictionary output (SSL models) and direct tensor output (MFCC/LFCC)
        if isinstance(features_output, dict):
            features = features_output['hidden_states'][-1]
        else:
            features = features_output

        lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16).to(DEVICE)

        # Forward pass
        outputs = forward_pass(model, features, lengths, dropout_prob)

        # Loss computation
        loss = criterion(outputs, labels)
        if torch.isnan(loss).any():
            print(f"NaN detected in loss during training")
            nan_count += torch.isnan(loss).sum().item()
            print(f"loss value: {loss.item()}")
            print(f"batch['file_name']: {batch['file_name']}")
            print(f"in train_one_epoch batch, nan_count: {nan_count}")
            continue

        epoch_loss += loss.item()

        # Backward and optimization
        backward_and_optimize(model, loss, optimizer, max_grad_norm)

        # Collect predictions for evaluation
        utterance_predictions.extend(outputs)
        utterance_labels.extend(labels)
        files_names.extend(batch['file_name'])

    print("===================================================")
    print(f'In Training loop, Total loss NAN count: {nan_count}')
    
    # Average epoch loss
    epoch_loss /= len(train_loader)
    
    # Compute Precision, Recall, and F1
    utterance_predictions_tensor = torch.cat(utterance_predictions)
    utterance_labels_tensor = torch.cat(utterance_labels)
    precision, recall, f1 = compute_precision_recall_f1(utterance_predictions_tensor, utterance_labels_tensor)
    
    return epoch_loss, utterance_predictions, utterance_labels, files_names, nan_count, precision, recall, f1

# ===========================================================================================================================

# def load_checkpoint(checkpoint_path, model, optimizer, DEVICE='cpu'):
#     """Load model and optimizer state from checkpoint"""
#     if not os.path.exists(checkpoint_path):
#         print(f"Checkpoint not found at {checkpoint_path}")
#         return 0
    
#     checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint.get('epoch', 0)
    
#     print(f"✓ Checkpoint loaded from epoch {epoch}")
#     return epoch


# def train_model(dataset_name,train_data_path, train_labels_path,dev_data_path, dev_labels_path, ssl_ckpt_path,apply_transform,
#                 save_feature_extractor=False,feature_dim=768, num_heads=8, hidden_dim=128, max_dropout=0.2,
#                 depthwise_conv_kernel_size=31, conformer_layers=1, max_pooling_factor=3,LEARNING_RATE=0.0001,
#                 BATCH_SIZE=32,NUM_EPOCHS=1, num_workers=0, prefetch_factor=None,
#                 monitor_dev_epoch=0,save_interval=float('inf'),
#                 model_save_path=os.path.join(os.getcwd(),'models/back_end_models'),
#                 patience=10,max_grad_norm=1.0,gamma=0.9,pin_memory=False,DEVICE='cpu'):

def train_model(config, dataset_name, train_data_path, train_labels_path,
                dev_data_path, dev_labels_path, apply_transform,
                save_feature_extractor=False, LEARNING_RATE=0.0001,
                BATCH_SIZE=32, NUM_EPOCHS=1, num_workers=0, prefetch_factor=None,
                monitor_dev_epoch=0, save_interval=float('inf'),
                model_save_path=os.path.join(os.getcwd(),'models/back_end_models'),
                patience=10, max_grad_norm=1.0, gamma=0.9, pin_memory=False, DEVICE='cpu'):


    """Train the model for NUM_EPOCHS"""
    # Initialize W&B
    initialize_wandb()

    # # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Initialize model, feature extractor, optimizer, loss function
    # PS_Model, feature_extractor, optimizer = initialize_models(ssl_ckpt_path, save_feature_extractor,
    #                   feature_dim, num_heads,hidden_dim,max_dropout,depthwise_conv_kernel_size,conformer_layers,max_pooling_factor, 
    #                   LEARNING_RATE,DEVICE)


    PS_Model, feature_extractor, optimizer = initialize_models(
        config, save_feature_extractor, LEARNING_RATE, DEVICE)


    criterion = initialize_loss_function().to(DEVICE)

    # Initialize data loader
    train_loader = initialize_data_loader(dataset_name,train_data_path, train_labels_path,BATCH_SIZE, True, num_workers, prefetch_factor,pin_memory,apply_transform)
    
    # Initialize learning rate scheduler (if enabled)
    use_lr_scheduler = config['training'].get('use_lr_scheduler', True)
    if use_lr_scheduler:
        LR_SCHEDULER = initialize_lr_scheduler(optimizer, config=config, use_scheduler=True)
        print(f"✓ Learning rate scheduler enabled: ReduceLROnPlateau")
    else:
        LR_SCHEDULER = None
        print(f"✓ Using fixed learning rate: {LEARNING_RATE}")

    wandb.watch(PS_Model, log_freq=100,log='all')

    # # checkpoint_full_path = os.path.join(model_save_path, resume_from_checkpoint)
    # checkpoint_full_path = "/root/Partial_Spoof_Detection_System/models/back_end_models/RFP_Dataset_model_epochs30_batch8_lr0.00075_20260104_054502.pth"
    # start_epoch = load_checkpoint(checkpoint_full_path, PS_Model, optimizer, DEVICE)

    # Set model to train
    PS_Model.train()
    total_train_nan_counter=0
    total_dev_nan_counter=0
    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):

        dropout_prob = adjust_dropout_prob(PS_Model, epoch, NUM_EPOCHS)

        # Training step for the current epoch
        epoch_loss, utterance_predictions, utterance_labels, files_names, train_nan_counter, train_precision, train_recall, train_f1 = train_one_epoch(
            PS_Model, train_loader, feature_extractor, optimizer, criterion, max_grad_norm, dropout_prob, DEVICE)

        total_train_nan_counter += train_nan_counter
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"model_epochs{epoch + 1}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pth"
            save_checkpoint(PS_Model, optimizer, epoch + 1, os.path.join(model_save_path, model_filename))

        # Compute and log metrics
        utterance_labels = torch.cat(utterance_labels)
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_metrics(utterance_predictions, utterance_labels)

        # Validation step (optional)
        if (epoch + 1) >= monitor_dev_epoch:
            # Initialize dev data loader
            dev_data_loader = initialize_data_loader(dataset_name, dev_data_path, dev_labels_path, BATCH_SIZE, False, num_workers, prefetch_factor, pin_memory)
 
            print(f"train_loader: {len(train_loader)} , dev_data_loader: {len(dev_data_loader)}")
            dev_metrics_dict, dev_nan_counter = dev_one_epoch(PS_Model, feature_extractor, criterion, dev_data_loader, 0, DEVICE)
            total_dev_nan_counter += dev_nan_counter

            # Log metrics to W&B with precision, recall, f1
            if save_feature_extractor:
                log_metrics_to_wandb(epoch, epoch_loss, utterance_eer, utterance_eer_threshold, optimizer.param_groups[1]['lr'], optimizer.param_groups[0]['lr'], 
                                   dropout_prob, dev_metrics_dict, train_precision, train_recall, train_f1)
            else:
                log_metrics_to_wandb(epoch, epoch_loss, utterance_eer, utterance_eer_threshold, optimizer.param_groups[0]['lr'], 0, 
                                   dropout_prob, dev_metrics_dict, train_precision, train_recall, train_f1)

            # Update learning rate scheduler if enabled
            if LR_SCHEDULER is not None:
                LR_SCHEDULER.step(dev_metrics_dict['utterance_eer'])
            
            # Early stopping check
            early_stopping(dev_metrics_dict['utterance_eer'], PS_Model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        else:
            if save_feature_extractor:
                log_metrics_to_wandb(epoch, epoch_loss, utterance_eer, utterance_eer_threshold, optimizer.param_groups[1]['lr'], optimizer.param_groups[0]['lr'], 
                                   dropout_prob, dev_metrics_dict=None, train_precision=train_precision, train_recall=train_recall, train_f1=train_f1)
            else:
                log_metrics_to_wandb(epoch, epoch_loss, utterance_eer, utterance_eer_threshold, optimizer.param_groups[0]['lr'], 0, 
                                   dropout_prob, dev_metrics_dict=None, train_precision=train_precision, train_recall=train_recall, train_f1=train_f1)
            
            # Update learning rate scheduler if enabled
            if LR_SCHEDULER is not None:
                LR_SCHEDULER.step()


    # Generate a unique filename based on hyperparameters
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"{dataset_name}_model_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pth"
    if save_feature_extractor:
        feature_extractor_filename = f"w2v_large_lv_fsh_swbd_cv_{timestamp}.pt"
        feature_extractor_save_path=os.path.join(model_save_path,feature_extractor_filename)
        save_checkpoint(feature_extractor, optimizer,NUM_EPOCHS,feature_extractor_save_path)

    # Save the trained model
    model_save_path=os.path.join(model_save_path,model_filename)
    save_checkpoint(PS_Model, optimizer,NUM_EPOCHS,model_save_path)
    print(f"Model saved to {model_save_path}")

    # # Save segment_predictions, segment_labels, utterance_predictions, utterance_labels
    # torch.save(utterance_predictions,os.path.join(os.getcwd(),f'outputs/utterance_predictions_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pt'))
    # torch.save(torch.tensor(utterance_labels),os.path.join(os.getcwd(),f'outputs/utterance_labels_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pt'))

    # # Save last metrics
    # training_metrics_dict=create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_loss)
    # training_metrics_dict_filename = f"metrics_dict_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.json"
    # training_metrics_dict_save_path=os.path.join(os.getcwd(),f'outputs/{training_metrics_dict_filename}')
    # save_json_dictionary(training_metrics_dict_save_path,training_metrics_dict)
    print("============================================================================================")
    print(f"total_train_nan_counter= {total_train_nan_counter} , total_dev_nan_counter= {total_dev_nan_counter}")
    if DEVICE=='cuda': torch.cuda.empty_cache()
    wandb.finish()
    print("Training complete!")




# def train(config=None):
#     """
#     Training function that accepts configuration
#     Args:
#         config: Configuration object from wandb or ConfigManager
#     """
#     if config is None:
#         # Initialize W&B if no config provided
#         initialize_wandb()
#         config = wandb.config
    
#     train_model(
#         dataset_name=config['data']['dataset_name'],
#         train_data_path=config['data']['train_data_path'],
#         train_labels_path=config['data']['train_labels_path'],
#         dev_data_path=config['data']['dev_data_path'],
#         dev_labels_path=config['data']['dev_labels_path'],
#         ssl_ckpt_path=config['paths']['ssl_checkpoint'],
#         apply_transform=config['system']['apply_transform'],
#         save_feature_extractor=config['system']['save_feature_extractor'],
#         feature_dim=config['model']['feature_dim'],
#         num_heads=config['model']['num_heads'],
#         hidden_dim=config['model']['hidden_dim'],
#         max_dropout=config['model']['max_dropout'],
#         depthwise_conv_kernel_size=config['model']['depthwise_conv_kernel_size'],
#         conformer_layers=config['model']['conformer_layers'],
#         max_pooling_factor=config['model']['max_pooling_factor'],
#         LEARNING_RATE=config['training']['learning_rate'],
#         BATCH_SIZE=config['training']['batch_size'],
#         NUM_EPOCHS=config['training']['num_epochs'],
#         num_workers=config['system']['num_workers'],
#         prefetch_factor=config['system']['prefetch_factor'],
#         pin_memory=config['system']['pin_memory'],
#         monitor_dev_epoch=config['training']['monitor_dev_epoch'],
#         save_interval=config['training']['save_interval'],
#         model_save_path=config['paths']['model_save_dir'],
#         patience=config['training']['patience'],
#         max_grad_norm=config['training']['max_grad_norm'],
#         gamma=config['training']['gamma'],
#         DEVICE=config['system']['device']
#     )

def train(config=None):
    """
    Training function that accepts configuration
    Args:
        config: Configuration object from wandb or ConfigManager
    """
    if config is None:
        initialize_wandb()
        config = wandb.config
    
    # ====== NEW: Log sequence model information ======
    sequence_model_type = config['model'].get('sequence_model_type', 'conformer')
    print(f"\n{'='*80}")
    print(f"Training with {sequence_model_type.upper()} sequence model")
    print(f"{'='*80}\n")
    # ====== END NEW ======


    train_model(
        config=config,  # Pass the full config
        dataset_name=config['data']['dataset_name'],
        train_data_path=config['data']['train_data_path'],
        train_labels_path=config['data']['train_labels_path'],
        dev_data_path=config['data']['dev_data_path'],
        dev_labels_path=config['data']['dev_labels_path'],
        apply_transform=config['system']['apply_transform'],
        save_feature_extractor=config['system']['save_feature_extractor'],
        LEARNING_RATE=config['training']['learning_rate'],
        BATCH_SIZE=config['training']['batch_size'],
        NUM_EPOCHS=config['training']['num_epochs'],
        num_workers=config['system']['num_workers'],
        prefetch_factor=config['system']['prefetch_factor'],
        pin_memory=config['system']['pin_memory'],
        monitor_dev_epoch=config['training']['monitor_dev_epoch'],
        save_interval=config['training']['save_interval'],
        model_save_path=config['paths']['model_save_dir'],
        patience=config['training']['patience'],
        max_grad_norm=config['training']['max_grad_norm'],
        gamma=config['training']['gamma'],
        DEVICE=config['system']['device']
    )


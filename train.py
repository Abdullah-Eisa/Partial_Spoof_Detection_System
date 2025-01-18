import os
import torch
import wandb
from torch.optim import lr_scheduler
from datetime import datetime
from tqdm import tqdm

import matplotlib.pyplot as plt

# from utils import initialize_wandb, initialize_models, initialize_loss_function, initialize_data_loader, save_checkpoint, compute_metrics, log_metrics_to_wandb
from utils import *
from preprocess import *
from model import *
from inference import dev_one_epoch

# ===========================================================================================================================
def train_one_epoch(model, train_loader, feature_extractor, optimizer, criterion, max_grad_norm, dropout_prob=0, DEVICE='cpu'):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    # utterance_eer, utterance_eer_threshold = 0, 0
    utterance_predictions = []
    utterance_labels = []
    files_names = []
    nan_count=0
    c=0
    for batch in tqdm(train_loader, desc="Train Batches", leave=False):
        # if c>8:
        #     break
        # else:
        #     c+=1
        waveforms = batch['waveform'].to(DEVICE)
        labels = batch['label'].to(DEVICE).unsqueeze(1).float()

        optimizer.zero_grad()

        # Feature extraction
        features = feature_extractor(waveforms)['hidden_states'][-1]
        lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16).to(DEVICE)

        # Forward pass
        outputs = forward_pass(model, features, lengths, dropout_prob)

        # Loss computation
        loss = criterion(outputs, labels)
        if torch.isnan(loss).any():
            print(f"NaN detected in loss during training")
            # c+=1
            nan_count+=torch.isnan(loss).sum().item()
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
    return epoch_loss, utterance_predictions, utterance_labels, files_names, nan_count

# ===========================================================================================================================
def train_model(train_data_path, train_labels_path,dev_data_path, dev_labels_path, ssl_ckpt_path,apply_transform,
                save_feature_extractor=False,feature_dim=768, num_heads=8, hidden_dim=128, max_dropout=0.2,
                depthwise_conv_kernel_size=31, conformer_layers=1, max_pooling_factor=3,LEARNING_RATE=0.0001,
                BATCH_SIZE=32,NUM_EPOCHS=1, num_workers=0, prefetch_factor=None,
                monitor_dev_epoch=0,save_interval=float('inf'),
                model_save_path=os.path.join(os.getcwd(),'models/back_end_models'),
                patience=10,max_grad_norm=1.0,gamma=0.9,pin_memory=False,DEVICE='cpu'):

    """Train the model for NUM_EPOCHS"""
    # Initialize W&B
    initialize_wandb()

    # # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Initialize model, feature extractor, optimizer, loss function
    PS_Model, feature_extractor, optimizer = initialize_models(ssl_ckpt_path, save_feature_extractor,
                      feature_dim, num_heads,hidden_dim,max_dropout,depthwise_conv_kernel_size,conformer_layers,max_pooling_factor, 
                      LEARNING_RATE,DEVICE)


    checkpoint = torch.load(os.path.join(os.getcwd(),'models/back_end_models/RFP_model_epochs30_batch8_lr0.0001_20250118_020114.pth'))
    print("loading RFP_model_epochs30_batch8_lr0.0001_20250118_020114.pth ...")
    PS_Model.load_state_dict(checkpoint['model_state_dict'])

    criterion = initialize_loss_function().to(DEVICE)

    # Initialize data loader
    train_loader = initialize_data_loader(train_data_path, train_labels_path,BATCH_SIZE, True, num_workers, prefetch_factor,pin_memory,apply_transform)
    # train_labels_dict= load_json_dictionary(train_labels_path)
    # train_labels_dict= load_labels_txt2dict(train_labels_path)

    LR_SCHEDULER = initialize_lr_scheduler(optimizer)

    wandb.watch(PS_Model, log_freq=100,log='all')
    # Set model to train
    PS_Model.train()
    total_train_nan_counter=0
    total_dev_nan_counter=0
    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):

        dropout_prob = adjust_dropout_prob(PS_Model, epoch, NUM_EPOCHS)

        # Training step for the current epoch
        epoch_loss, utterance_predictions, utterance_labels, files_names ,train_nan_counter = train_one_epoch(
            PS_Model, train_loader, feature_extractor, optimizer, criterion,max_grad_norm,dropout_prob, DEVICE)

        total_train_nan_counter+=train_nan_counter
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"model_epochs{epoch + 1}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pth"
            save_checkpoint(PS_Model, optimizer, epoch + 1, os.path.join(model_save_path, model_filename))

        # Compute and log metrics
        # utterance_labels = torch.tensor([train_labels_dict[file_name] for file_name in files_names])
        utterance_labels = torch.cat(utterance_labels)
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_metrics(utterance_predictions, utterance_labels)

        # Validation step (optional)
        if (epoch + 1) >= monitor_dev_epoch:
            dev_data_loader=initialize_data_loader(dev_data_path, dev_labels_path,BATCH_SIZE,False,num_workers, prefetch_factor,pin_memory)
            # dev_labels_dict= load_json_dictionary(dev_labels_path)
            # dev_labels_dict= load_labels_txt2dict(dev_labels_path)
            print(f"train_loader: {len(train_loader)} , dev_data_loader: {len(dev_data_loader)}")
            # dev_metrics_dict, dev_nan_counter = dev_one_epoch(PS_Model, feature_extractor,criterion,dev_data_loader, dev_labels_dict,0,DEVICE)
            dev_metrics_dict, dev_nan_counter = dev_one_epoch(PS_Model, feature_extractor,criterion,dev_data_loader,0,DEVICE)
            total_dev_nan_counter+=dev_nan_counter

            if save_feature_extractor:
                log_metrics_to_wandb(epoch, epoch_loss, utterance_eer, utterance_eer_threshold, optimizer.param_groups[1]['lr'], optimizer.param_groups[0]['lr'],dropout_prob, dev_metrics_dict)               # Log metrics to W&B
            else:
                log_metrics_to_wandb(epoch, epoch_loss, utterance_eer, utterance_eer_threshold,optimizer.param_groups[0]['lr'], 0,dropout_prob, dev_metrics_dict)               # Log metrics to W&B

            LR_SCHEDULER.step(dev_metrics_dict['utterance_eer'])
            
            # Early stopping check
            early_stopping(dev_metrics_dict['utterance_eer'], PS_Model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        else:
            if save_feature_extractor:
                log_metrics_to_wandb(epoch, epoch_loss, utterance_eer, utterance_eer_threshold, optimizer.param_groups[1]['lr'], optimizer.param_groups[0]['lr'], dropout_prob, dev_metrics_dict= None)  
            else:       # Log metrics to W&B
                log_metrics_to_wandb(epoch, epoch_loss, utterance_eer, utterance_eer_threshold, optimizer.param_groups[0]['lr'], 0, dropout_prob, dev_metrics_dict= None)    
                     # Log metrics to W&B
            LR_SCHEDULER.step()


    # Generate a unique filename based on hyperparameters
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"RFP_model_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pth"
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




def train():
    # Initialize W&B
    # wandb.init(project='partial_spoof_Wav2Vec2_Conformer_binary_classifier')
    initialize_wandb()

    # Extract parameters from W&B configuration
    config = wandb.config
    
    # Get Device
    use_cuda= True
    use_cuda =  use_cuda and torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    print(f'device: {DEVICE}')

    pin_memory= True if DEVICE=='cuda' else False   # Enable page-locked memory for faster data transfer to GPU
    # Define training files and labels
    train_data_path=os.path.join(os.getcwd(),'database/RFP/training')
    # train_data_path=os.path.join(os.getcwd(),'database/mini_database/train')
    train_labels_path=os.path.join(os.getcwd(),'database/RFP/labels/ASVspoof2017_V2_train.trl.txt')

    dev_data_path=os.path.join(os.getcwd(), 'database/RFP/validation')
    # dev_data_path=os.path.join(os.getcwd(), 'database/mini_database/dev')
    dev_labels_path=os.path.join(os.getcwd(), 'database/RFP/labels/ASVspoof2017_V2_dev.trl.txt') 
    ssl_ckpt_path=os.path.join(os.getcwd(), 'models/w2v_large_lv_fsh_swbd_cv.pt')
    
    # Call train_model with parameters from W&B sweep
    train_model(train_data_path=train_data_path, 
               train_labels_path=train_labels_path,
               dev_data_path=dev_data_path, 
               dev_labels_path=dev_labels_path, 
               ssl_ckpt_path=ssl_ckpt_path,
               apply_transform=False,
               save_feature_extractor=False,
               feature_dim=768, 
               num_heads=8, 
               hidden_dim=128, 
               max_dropout=0.2625,
               depthwise_conv_kernel_size=31, 
               conformer_layers=1, 
               max_pooling_factor=3,
               LEARNING_RATE=config.LEARNING_RATE,
               BATCH_SIZE=config.BATCH_SIZE,
               NUM_EPOCHS=config.NUM_EPOCHS, 
               num_workers=8, 
               prefetch_factor=2,
               pin_memory=pin_memory,
               monitor_dev_epoch=0,
               save_interval=5,
               model_save_path=os.path.join(os.getcwd(),'models/back_end_models'),
               patience=15,
               max_grad_norm=1.0,
               gamma=0.9,
               DEVICE=DEVICE)
    




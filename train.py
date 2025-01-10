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
def train_one_epoch(model, train_loader, optimizer, criterion, max_grad_norm, DEVICE='cpu'):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    # utterance_eer, utterance_eer_threshold = 0, 0
    utterance_predictions = []
    files_names = []
    c=0
    for batch in tqdm(train_loader, desc="Train Batches", leave=False):
        if c>2:
            break
        else:
            c+=1
        fbank = batch['fbank'].to(DEVICE)
        labels = batch['label'].to(DEVICE).unsqueeze(1).float()

        optimizer.zero_grad()

        # Forward pass
        outputs = forward_pass(model,fbank)
        print(f"outputs: {outputs}")
        # Loss computation
        loss = criterion(outputs, labels)
        if torch.isnan(loss).any():
            print(f"NaN detected in loss during training")
            continue

        epoch_loss += loss.item()

        # Backward and optimization
        backward_and_optimize(model, loss, optimizer, max_grad_norm)

        # Collect predictions for evaluation
        utterance_predictions.extend(outputs)
        files_names.extend(batch['file_name'])

    # Average epoch loss
    epoch_loss /= len(train_loader)
    return epoch_loss, utterance_predictions, files_names

# ===========================================================================================================================
def train_model(train_data_path, train_labels_path,train_audio_conf,dev_data_path, dev_labels_path,dev_audio_conf,
                input_fdim,input_tdim,imagenet_pretrain, audioset_pretrain, model_size,
                LEARNING_RATE=0.0001,BATCH_SIZE=32,NUM_EPOCHS=1, num_workers=0, prefetch_factor=None,
                monitor_dev_epoch=0,save_interval=float('inf'),
                model_save_path=os.path.join(os.getcwd(),'models/back_end_models'),
                patience=10,max_grad_norm=1.0,milestones=[10,20,30],gamma=0.9,pin_memory=False,DEVICE='cpu'):

    """Train the model for NUM_EPOCHS"""
    # Initialize W&B
    initialize_wandb()

    # # Initialize early stopping
    # early_stopping = EarlyStopping(patience=patience, verbose=True)

    AST_model, optimizer= initialize_models(input_fdim,input_tdim, 
                            imagenet_pretrain, audioset_pretrain, 
                            model_size,LEARNING_RATE,DEVICE)

    criterion = initialize_loss_function().to(DEVICE)

    # Initialize data loader
    train_loader = initialize_data_loader(train_data_path, train_labels_path,train_audio_conf,BATCH_SIZE, True, num_workers, prefetch_factor,pin_memory)
    train_labels_dict= load_json_dictionary(train_labels_path)

    LR_SCHEDULER = initialize_lr_scheduler(optimizer,milestones, gamma)

    # wandb.watch(AST_model, log_freq=100,log='all')           # Log model gradients and parameters            ????????????????????????????????????????????
    # Set model to train
    AST_model.train()

    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):

        # Training step for the current epoch
        epoch_loss, utterance_predictions, files_names = train_one_epoch(
            AST_model, train_loader, optimizer, criterion,max_grad_norm, DEVICE)

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"model_epochs{epoch + 1}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pth"
            save_checkpoint(AST_model, optimizer, epoch + 1, os.path.join(model_save_path, model_filename))

        # Compute and log metrics
        utterance_labels = torch.tensor([train_labels_dict[file_name] for file_name in files_names])
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_metrics(utterance_predictions, utterance_labels)

        # Validation step (optional)
        if (epoch + 1) >= monitor_dev_epoch:
            dev_data_loader=initialize_data_loader(dev_data_path, dev_labels_path,dev_audio_conf,BATCH_SIZE,False,num_workers, prefetch_factor,pin_memory)
            dev_labels_dict= load_json_dictionary(dev_labels_path)
            # print(f"train_loader: {len(train_loader)} , dev_data_loader: {len(dev_data_loader)}")
            dev_metrics_dict = dev_one_epoch(AST_model,dev_data_loader, dev_labels_dict,criterion,0,DEVICE)
            
            # if save_feature_extractor:
            #     log_metrics_to_wandb(epoch, epoch_loss, utterance_eer, utterance_eer_threshold,optimizer.param_groups[0]['lr'],optimizer.param_groups[1]['lr'],dropout_prob, dev_metrics_dict)               # Log metrics to W&B
            # else:
            log_metrics_to_wandb(epoch, epoch_loss, utterance_eer, utterance_eer_threshold,optimizer.param_groups[0]['lr'],0,0, dev_metrics_dict)               # Log metrics to W&B

            LR_SCHEDULER.step(dev_metrics_dict['utterance_eer'])
            # Early stopping check
            # early_stopping(dev_metrics_dict['epoch_loss'], PS_Model)
            # if early_stopping.early_stop:
            #     print(f"Early stopping at epoch {epoch+1}")
            #     break

        else:
            log_metrics_to_wandb(epoch, epoch_loss, utterance_eer, utterance_eer_threshold,optimizer.param_groups[0]['lr'],optimizer.param_groups[1]['lr'],0, dev_metrics_dict= None)         # Log metrics to W&B
            LR_SCHEDULER.step()


    # Generate a unique filename based on hyperparameters
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"model_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pth"

    # Save the trained model
    model_save_path=os.path.join(model_save_path,model_filename)
    save_checkpoint(AST_model, optimizer,NUM_EPOCHS,model_save_path)
    print(f"Model saved to {model_save_path}")

    # # Save last metrics
    # training_metrics_dict=create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_loss)
    # training_metrics_dict_filename = f"metrics_dict_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.json"
    # training_metrics_dict_save_path=os.path.join(os.getcwd(),f'outputs/{training_metrics_dict_filename}')
    # save_json_dictionary(training_metrics_dict_save_path,training_metrics_dict)

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
    train_data_path=os.path.join(os.getcwd(),'database/train/con_wav')
    # train_data_path=os.path.join(os.getcwd(),'database/mini_database/train')
    train_labels_path=os.path.join(os.getcwd(),'database/utterance_labels/PartialSpoof_LA_cm_train_trl.json')
    dev_data_path=os.path.join(os.getcwd(), 'database/dev/con_wav')
    # dev_data_path=os.path.join(os.getcwd(), 'database/mini_database/dev')
    dev_labels_path=os.path.join(os.getcwd(), 'database/utterance_labels/PartialSpoof_LA_cm_dev_trl.json') 
    
    train_audio_conf = {
        'num_mel_bins': 128,
        'freqm': 48,  # frequency masking parameter
        'timem': 192,  # time masking parameter
        # 'mixup': 0,  # mix-up rate
        # 'dataset': 'Audioset',  # Dataset type
        # 'mean': 0.5,  # Mean value for normalization
        # 'std': 0.25,  # Standard deviation for normalization
        # 'skip_norm': False,  # Do not skip normalization
        # 'noise': True,  # Apply noise augmentation
        'target_length': 1024,  # Target length for spectrogram
    }
    dev_audio_conf = {
        'num_mel_bins': 128,
        'freqm': 0,  # frequency masking parameter
        'timem': 0,  # time masking parameter
        # 'mixup': 0,  # mix-up rate
        # 'dataset': 'Audioset',  # Dataset type
        # 'mean': 0.5,  # Mean value for normalization
        # 'std': 0.25,  # Standard deviation for normalization
        # 'skip_norm': False,  # Do not skip normalization
        # 'noise': True,  # Apply noise augmentation
        'target_length': 1024,  # Target length for spectrogram
    }
    # Call train_model with parameters from W&B sweep
    train_model(train_data_path=train_data_path, 
               train_labels_path=train_labels_path,
               train_audio_conf= train_audio_conf,
               dev_data_path=dev_data_path, 
               dev_labels_path=dev_labels_path, 
               dev_audio_conf= dev_audio_conf,
               input_fdim=128,
               input_tdim=1024,
               imagenet_pretrain=False, 
               audioset_pretrain=False, 
               model_size='base384',
               LEARNING_RATE=config.LEARNING_RATE,
               BATCH_SIZE=config.BATCH_SIZE,
               NUM_EPOCHS=config.NUM_EPOCHS, 
               num_workers=8, 
               prefetch_factor=2,
               pin_memory=pin_memory,
               monitor_dev_epoch=0,
               save_interval=10,
               model_save_path=os.path.join(os.getcwd(),'models/back_end_models'),
               patience=10,
               max_grad_norm=1.0,
               milestones=[10,15,20,25,30],
               gamma=0.9,
               DEVICE=DEVICE)
    



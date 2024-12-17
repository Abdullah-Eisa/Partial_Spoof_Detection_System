
import os
from tqdm import tqdm  # Correctly import tqdm
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# from transformers import Wav2Vec2Processor, 
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model

import wandb

from utils import *
from model import *
from model import MyUpdatedSpoofingDetectionModel

from inference import dev_model


# ... (your training and inference functions)

def train_model(train_directory, train_labels_dict, 
                BATCH_SIZE=32, NUM_EPOCHS=1,LEARNING_RATE=0.0001,
                model_save_path=os.path.join(os.getcwd(),'models/back_end_models'),
                DEVICE='cpu',save_interval=float('inf'),patience=10,save_feature_extractor=False,max_grad_norm=1.0,monitor_dev_epoch=50):

    # Initialize W&B
    wandb.init(project='partial_spoof_trial_2')

    # # Initialize early stopping
    # early_stopping = EarlyStopping(patience=patience, verbose=True)

    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    # Ensure the model save path exists
    os.makedirs(model_save_path, exist_ok=True)
    # Load utterance labels
    BASE_DIR = os.getcwd()
    PartialSpoof_LA_cm_train_trl_dict_path = os.path.join(BASE_DIR,'database/utterance_labels/PartialSpoof_LA_cm_train_trl.json')
    PartialSpoof_LA_cm_train_trl_dict= load_json_dictionary(PartialSpoof_LA_cm_train_trl_dict_path)

    # Load feature extractor
    ssl_ckpt_path = os.path.join(os.getcwd(), 'models/w2v_large_lv_fsh_swbd_cv.pt')
    feature_extractor = torch.hub.load('s3prl/s3prl', 'wav2vec2', model_path=ssl_ckpt_path).to(DEVICE)

    # Initialize the model, loss function, and optimizer
    hidd_dims ={'wav2vec2':768, 'wav2vec2_large':1024}
    PS_Model = MyUpdatedSpoofingDetectionModel(feature_dim=hidd_dims['wav2vec2'], num_heads=8, hidden_dim=128, num_classes=33,conformer_layers=1).to(DEVICE)  # Move model to the configured device

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        PS_Model = nn.DataParallel(PS_Model).to(DEVICE)
        print("Parallelizing model on ", torch.cuda.device_count(), "GPUs!")

    if save_feature_extractor:
        # Freeze all layers except the last one (final_proj)
        for name, param in feature_extractor.named_parameters():
            if 'final_proj' not in name:  # Check if the layer is not the last one
                param.requires_grad = False
            else:
                param.requires_grad = True

        # optimizer = optim.Adam(list(PS_Model.parameters()) + list(wav2vec2_model.parameters()), lr=LEARNING_RATE)
        optimizer = optim.AdamW([
            {'params': feature_extractor.parameters(), 'lr': LEARNING_RATE / 5} ,
            {'params': PS_Model.parameters()}], lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)

    else:
        # optimizer = optim.Adam(PS_Model.parameters(), lr=LEARNING_RATE)
        optimizer = optim.AdamW(PS_Model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
        feature_extractor.eval()
        


    # criterion = nn.BCELoss()  # Binary Cross Entropy Loss for multi-label classification
    criterion = CustomLoss().to(DEVICE)


    gamma=0.9
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # Get the data loader
    train_loader = get_raw_labeled_audio_data_loaders(train_directory, train_labels_dict,batch_size=BATCH_SIZE, shuffle=True, num_workers=8, prefetch_factor=2)


    PS_Model,_,_=load_checkpoint(PS_Model, optimizer, path=os.path.join(os.getcwd(),'models/back_end_models/model_epochs15_batch8_lr0.01_20241214_171142.pth'))

    # Logging gradients with wandb.watch
    wandb.watch(PS_Model, log_freq=100,log='all')

    PS_Model.train()  # Set the model to training mode

    files_names=[]
    training_segment_eer_per_epoch=[]
    dev_segment_eer_per_epoch=[]

    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
        PS_Model.train()  # Set the model to training mode

        # Adjust dropout probability for the current epoch
        dropout_prob = PS_Model.adjust_dropout(epoch, NUM_EPOCHS)

        epoch_loss = 0
        utterance_eer, utterance_eer_threshold=0,0
        segment_eer, segment_eer_threshold=0,0
        utterance_predictions=[]
        # utterance_pooling_predictions=[]
        segment_predictions=[]
        segment_labels=[]
        # c=0
        for batch in tqdm(train_loader, desc="Train Batches", leave=False):
            # if c>16:
            #     break
            # else:
            #     c+=1
            waveforms = batch['waveform'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass through wav2vec2 for feature extraction
            features = feature_extractor(waveforms)['hidden_states'][-1] 
            # print(f'type {type(features)}  with size {features.size()} , features= {features}')

            # lengths should be the number of non-padded frames in each sequence
            lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16).to(DEVICE)  # (batch_size,)

            # Pass features to model and get predictions
            outputs = PS_Model(features,lengths,dropout_prob)
            # outputs = PS_Model(features,lengths)
            # print(f"PS_Model outputs with size: {logits.size()}")
            # outputs = torch.argmax(logits, dim=-1)
            # print(f"predicted_class_ids: {outputs}")

            # Calculate loss
            loss = criterion(outputs, labels)  
            epoch_loss += loss.item()


            # Backward pass and optimization
            loss.backward()
            # Apply gradient clipping to prevent vanishing/exploding gradients
            torch.nn.utils.clip_grad_norm_(PS_Model.parameters(), max_grad_norm)
    
            optimizer.step()


            with torch.no_grad():  # No need to compute gradients for EER calculation
                # Calculate utterance predictions
                utterance_predictions.extend(get_uttEER_by_seg(outputs,labels))
                segment_predictions.extend(outputs)
                segment_labels.extend(labels)
                # Accumulate files names
                files_names.extend(batch['file_name'])



        # Save checkpoint
        if NUM_EPOCHS>=save_interval and (epoch + 1) % (save_interval) == 0:
            # Generate a unique filename based on hyperparameters
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"model_epochs{epoch + 1}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pth"
                        
            save_checkpoint(PS_Model, optimizer, epoch + 1,os.path.join(model_save_path,model_filename))


        # Get Average Utterance EER for the epoch
        utterance_labels =[PartialSpoof_LA_cm_train_trl_dict[file_name] for file_name in files_names]
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_eer(utterance_predictions,torch.tensor(utterance_labels))
        # utterance_pooling_predictions = torch.cat(utterance_pooling_predictions, dim=0)
        # utterance_pooling_eer, utterance_pooling_eer_threshold = compute_eer(utterance_pooling_predictions,torch.tensor(utterance_labels))

        # Calculate Training segment EER
        segment_predictions=torch.cat(segment_predictions, dim=0)
        segment_labels=torch.cat(segment_labels, dim=0)
        segment_eer, segment_eer_threshold = compute_eer(segment_predictions,segment_labels)

        # Average  loss for the epoch
        epoch_loss /= len(train_loader)



        if (epoch+1) >= monitor_dev_epoch :

            BASE_DIR = os.getcwd()
            # Define development files and labels
            dev_files_path=os.path.join(BASE_DIR,'database/dev/con_wav')
            # dev_files_path=os.path.join(BASE_DIR,'database/mini_database/dev')
            dev_seglab_64_path=os.path.join(BASE_DIR,'database/segment_labels/dev_seglab_0.64.npy')
            dev_seglab_64_dict = np.load(dev_seglab_64_path, allow_pickle=True).item()

            dev_metrics_dict=dev_model( PS_Model,dev_files_path, dev_seglab_64_dict,feature_extractor,dropout_prob, BATCH_SIZE,DEVICE=DEVICE)
            dev_segment_eer_per_epoch.append(dev_metrics_dict['segment_eer'])

            wandb.log({'epoch': epoch+1,'training_loss_epoch': epoch_loss,
                'training_segment_eer_epoch': segment_eer, 
                'training_segment_eer_threshold_epoch': segment_eer_threshold,
                'training_utterance_eer_epoch': utterance_eer,
                'training_utterance_eer_threshold_epoch': utterance_eer_threshold, 
                # 'training_utterance_pooling_eer_epoch': utterance_pooling_eer,
                # 'training_utterance_pooling_eer_threshold_epoch': utterance_pooling_eer_threshold, 
                'validation_loss_epoch': dev_metrics_dict['epoch_loss'],
                'validation_segment_eer_epoch': dev_metrics_dict['segment_eer'], 
                'validation_segment_eer_threshold_epoch': dev_metrics_dict['segment_eer_threshold'],
                'validation_utterance_eer_epoch': dev_metrics_dict['utterance_eer'],
                'validation_utterance_eer_threshold_epoch': dev_metrics_dict['utterance_eer_threshold']                      
                })
        else:
            wandb.log({'epoch': epoch+1,'training_loss_epoch': epoch_loss,
                'training_segment_eer_epoch': segment_eer, 
                'training_segment_eer_threshold_epoch': segment_eer_threshold,
                'training_utterance_eer_epoch': utterance_eer,
                'training_utterance_eer_threshold_epoch': utterance_eer_threshold                  
                })

        # Early stopping check
        # early_stopping(dev_metrics_dict['epoch_loss'], PS_Model)
        # if early_stopping.early_stop:
        #     print(f"Early stopping at epoch {epoch+1}")
        #     break

        scheduler.step()


    # Generate a unique filename based on hyperparameters
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"model_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pth"
    if save_feature_extractor:
        feature_extractor_filename = f"w2v_large_lv_fsh_swbd_cv_{timestamp}.pt"
        feature_extractor_save_path=os.path.join(model_save_path,feature_extractor_filename)
        save_checkpoint(feature_extractor, optimizer,NUM_EPOCHS,feature_extractor_save_path)

    # Save the trained model
    model_save_path=os.path.join(model_save_path,model_filename)
    save_checkpoint(PS_Model, optimizer,NUM_EPOCHS,model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save segment_predictions, segment_labels, utterance_predictions, utterance_labels
    torch.save(segment_predictions,os.path.join(os.getcwd(),f'outputs/segment_predictions_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pt'))
    torch.save(utterance_predictions,os.path.join(os.getcwd(),f'outputs/utterance_predictions_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pt'))
    torch.save(segment_labels,os.path.join(os.getcwd(),f'outputs/segment_labels_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pt'))
    torch.save(torch.tensor(utterance_labels),os.path.join(os.getcwd(),f'outputs/utterance_labels_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pt'))

    # Save last metrics
    training_metrics_dict=create_metrics_dict(utterance_eer,utterance_eer_threshold,segment_eer,segment_eer_threshold,epoch_loss)
    training_metrics_dict_filename = f"metrics_dict_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.json"
    training_metrics_dict_save_path=os.path.join(os.getcwd(),f'outputs/{training_metrics_dict_filename}')
    save_json_dictionary(training_metrics_dict_save_path,training_metrics_dict)

    if DEVICE=='cuda': torch.cuda.empty_cache()
    wandb.finish()
    print("Training complete!")



def train():
    # Initialize W&B
    wandb.init(project='partial_spoof_trial_2')

    # Extract parameters from W&B configuration
    config = wandb.config
    
    # Get Device
    use_cuda= True
    use_cuda =  use_cuda and torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    print(f'device: {DEVICE}')

    # Define your paths and other fixed arguments
    BASE_DIR = os.getcwd()

    # Define training files and labels
    train_files_path=os.path.join(BASE_DIR,'database/train/con_wav')
    # train_files_path=os.path.join(BASE_DIR,'database/mini_database/train3')
    train_seglab_64_path=os.path.join(BASE_DIR,'database/segment_labels/train_seglab_0.64.npy')
    train_seglab_64_dict = np.load(train_seglab_64_path, allow_pickle=True).item()


    # Call train_model with parameters from W&B sweep
    train_model(
        train_directory=train_files_path,
        train_labels_dict=train_seglab_64_dict,
        BATCH_SIZE=config.BATCH_SIZE,
        NUM_EPOCHS=config.NUM_EPOCHS,
        LEARNING_RATE=config.LEARNING_RATE,
        DEVICE=DEVICE,
        save_interval=10
    )




if __name__ == "__main__":

    # Device configuration
    use_cuda= True
    use_cuda =  use_cuda and torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    print(f'device: {DEVICE}')

    BASE_DIR = os.getcwd()

    # Define training files and labels
    # train_files_path=os.path.join(BASE_DIR,'database/mini_database/train')
    # train_files_path=os.path.join(BASE_DIR,'database/mini_database/train2')
    train_files_path=os.path.join(BASE_DIR,'database/mini_database/train3')
    train_seglab_64_path=os.path.join(BASE_DIR,'database/segment_labels/train_seglab_0.64.npy')
    train_seglab_64_dict = np.load(train_seglab_64_path, allow_pickle=True).item()


    # Load the tokenizer and model from the local directory
    # Wav2Vec2_tokenizer = Wav2Vec2Tokenizer.from_pretrained("models/local_wav2vec2_tokenizer")
    # # Wav2Vec2_model = Wav2Vec2Model.from_pretrained("models/local_wav2vec2_model")
    # Wav2Vec2_model = Wav2Vec2Model.from_pretrained("models/local_wav2vec2_model").to(DEVICE)
    # Wav2Vec2_model.eval()


    # Record the start time
    start_time = datetime.now()
    # train model
    train_model(train_files_path, train_seglab_64_dict,  BATCH_SIZE=16,NUM_EPOCHS=5,DEVICE=DEVICE)

    # Record the end time
    end_time = datetime.now()
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time}")

    # Extract hours, minutes, and seconds
    total_seconds = total_training_time.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    # Print training time in hours, minutes, and seconds
    print(f"Total training time: {hours} hours, {minutes} minutes, {seconds} seconds")


    # file_name='CON_T_0000000'
    # print(train_seglab_64_dict[file_name])
    # print(type(train_seglab_64_dict[file_name]))
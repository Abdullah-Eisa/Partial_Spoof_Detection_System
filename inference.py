
import os
from tqdm import tqdm  
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim


from utils import *
from utils import load_json_dictionary
from model import *
from preprocess import initialize_data_loader ,initialize_loss_function


import os
import torch
import wandb
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torchaudio.models as tam
import math

import torchaudio
from torch.utils.data import Dataset, DataLoader , ConcatDataset
# from transformers import Wav2Vec2Processor, 
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve

from torch.nn.utils.rnn import pad_sequence

import torch.multiprocessing as mp


# ===========================================================================================================================
def inference_helper(model, feature_extractor,criterion,
                  test_data_loader, test_labels_dict,DEVICE='cpu'):
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
    dropout_prob=0
    # c=0
    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc="Test Batches", leave=False):
            # if c>8:
            #     break
            # else:
            #     c+=1
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
                continue
            epoch_loss += loss.item()

            with torch.no_grad():
                # Calculate utterance predictions
                utterance_predictions.extend(outputs)
                # Accumulate files names
                files_names.extend(batch['file_name'])


        # Get Average Utterance EER for the epoch
        utterance_labels =torch.tensor([test_labels_dict[file_name] for file_name in files_names])
        # print(f'epoch {epoch} , utterance_labels: {utterance_labels}')
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_metrics(utterance_predictions,utterance_labels)

        # Average loss for the epoch
        epoch_loss /= len(test_data_loader)


    # Print epoch testing results
    print(f'Testing/Inference Complete. Test Loss: {epoch_loss:.4f},\n'
               f'Average Test Utterance EER: {utterance_eer:.4f}, Average Test Utterance EER Threshold: {utterance_eer_threshold:.4f}')

    return create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_loss)





def inference(eval_data_path=os.path.join(os.getcwd(),'database/eval/con_wav'),
    eval_labels_path = os.path.join(os.getcwd(),'database/utterance_labels/PartialSpoof_LA_cm_eval_trl.json'),
    ssl_ckpt_path=os.path.join(os.getcwd(), 'models/w2v_large_lv_fsh_swbd_cv.pt'),
    BATCH_SIZE=16, num_workers=0, prefetch_factor=None, DEVICE='cpu'):

    print("infer_model is working ... ")
    # Get Device
    use_cuda= True
    use_cuda =  use_cuda and torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    print(f'inference device: {DEVICE}')

    # Define your paths and other fixed arguments
    BASE_DIR = os.getcwd()

    # Define training files and labels
    # eval_data_path=os.path.join(BASE_DIR,'database/eval/con_wav')
    # eval_labels_path = os.path.join(BASE_DIR,'database/utterance_labels/PartialSpoof_LA_cm_eval_trl.json')
    eval_labels_dict= load_json_dictionary(eval_labels_path)
    pin_memory= True if DEVICE=='cuda' else False   # Enable page-locked memory for faster data transfer to GPU
    eval_data_loader = initialize_data_loader(eval_data_path, eval_labels_path,BATCH_SIZE,False, num_workers, prefetch_factor,pin_memory)


    # Load feature extractor
    # ssl_ckpt_path = os.path.join(os.getcwd(), 'models/w2v_large_lv_fsh_swbd_cv.pt')
    # ssl_ckpt_name='w2v_large_lv_fsh_swbd_cv_20241223_152156.pt'
    # ssl_ckpt_path = os.path.join(os.getcwd(), f'models/back_end_models/{ssl_ckpt_name}')
    feature_extractor = torch.hub.load('s3prl/s3prl', 'wav2vec2', model_path=ssl_ckpt_path).to(DEVICE)
    feature_extractor.eval()

    # Initialize Binary Spoofing Classification Model
    PS_Model_name='model_epochs60_batch8_lr0.005_20241226_214707.pth'
    model_path=os.path.join(os.getcwd(),f'models/back_end_models/{PS_Model_name}')
    PS_Model = BinarySpoofingClassificationModel(feature_dim=768,
                                                num_heads=8,
                                                hidden_dim=128, 
                                                max_dropout=0, 
                                                depthwise_conv_kernel_size=31, 
                                                conformer_layers=1, 
                                                max_pooling_factor=3).to(DEVICE)
    # PS_Model,_,_=load_checkpoint(PS_Model, optimizer, path=os.path.join(os.getcwd(),'models/back_end_models/model_epochs30_batch8_lr0.005_20241216_013405.pth'))
    checkpoint = torch.load(model_path)
    PS_Model.load_state_dict(checkpoint['model_state_dict'])
    PS_Model.eval()  # Set the model to evaluation mode


    criterion = initialize_loss_function().to(DEVICE)


    inference_helper(
        model=PS_Model,
        feature_extractor=feature_extractor,
        criterion=criterion,
        test_data_loader=eval_data_loader, 
        test_labels_dict=eval_labels_dict,
        DEVICE=DEVICE)

    if DEVICE=='cuda': torch.cuda.empty_cache()


# ===========================================================================================================================
def dev_one_epoch(model, feature_extractor,criterion,
                  dev_data_loader, dev_labels_dict,dropout_prob=0,DEVICE='cpu'):
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
    # c=0
    with torch.no_grad():
        for batch in tqdm(dev_data_loader, desc="Dev Batches", leave=False):
            # if c>8:
            #     break
            # else:
            #     c+=1
            waveforms = batch['waveform'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            labels = labels.unsqueeze(1).float()   # Converts labels from shape [batch_size] to [batch_size, 1]

            # Forward pass through wav2vec2 for feature extraction
            features = feature_extractor(waveforms)['hidden_states'][-1] 
             # print(f'type {type(features)}  with size {features.size()} , features= {features}')

            # lengths should be the number of non-padded frames in each sequence
            lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16).to(DEVICE)  # (batch_size,)

            # Pass features to model and get predictions
            # outputs = PS_Model(features,lengths,dropout_prob)
            outputs = forward_pass(model, features, lengths, dropout_prob)

            # Calculate loss
            loss = criterion(outputs, labels) 
            if torch.isnan(loss).any(): 
                print(f"NaN detected in validation loop loss") 
                continue
            epoch_loss += loss.item()

            with torch.no_grad():
                # Calculate utterance predictions
                utterance_predictions.extend(outputs)
                # Accumulate files names
                files_names.extend(batch['file_name'])


        # Get Average Utterance EER for the epoch
        utterance_labels =torch.tensor([dev_labels_dict[file_name] for file_name in files_names])
        # print(f'epoch {epoch} , utterance_labels: {utterance_labels}')
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_metrics(utterance_predictions,utterance_labels)

        # Average loss for the epoch
        epoch_loss /= len(dev_data_loader)


    # Print epoch dev progress
    # print(f'Epoch [{epoch + 1}] Complete. Validation Loss: {epoch_loss:.4f},\n'
    #            f'Average Validation Segment EER: {segment_eer:.4f}, Average Validation Segment EER Threshold: {segment_eer_threshold:.4f},\n'
    #            f'Average Validation Utterance EER: {utterance_eer:.4f}, Average Validation Utterance EER Threshold: {utterance_eer_threshold:.4f}')

    return create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_loss)



if __name__ == "__main__":
    print("SA, inference.py file !")
    use_cuda= True
    use_cuda =  use_cuda and torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    print(f'device: {DEVICE}')
    # Record the start time
    start_time = datetime.now()

    # inference()
    inference(eval_data_path=os.path.join(os.getcwd(),'database/eval/con_wav'),
        eval_labels_path = os.path.join(os.getcwd(),'database/utterance_labels/PartialSpoof_LA_cm_eval_trl.json'),
        ssl_ckpt_path=os.path.join(os.getcwd(), 'models/back_end_models/w2v_large_lv_fsh_swbd_cv_20241226_214707.pt'),
        BATCH_SIZE=16, num_workers=0, prefetch_factor=None, DEVICE=DEVICE)
    # Record the end time
    end_time = datetime.now()
    total_time = end_time - start_time
    print(f"Total time: {total_time}")

    # Extract hours, minutes, and seconds
    total_seconds = total_time.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    # Print training time in hours, minutes, and seconds
    print(f"Total time: {hours} hours, {minutes} minutes, {seconds} seconds")

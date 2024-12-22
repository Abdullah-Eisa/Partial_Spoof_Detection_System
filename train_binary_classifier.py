
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
# from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model

import wandb

# from utils import *
from utils import SelfWeightedPooling , load_json_dictionary ,load_checkpoint,save_checkpoint,compute_eer
# from model import *
# from model import MyUpdatedSpoofingDetectionModel

# from inference import dev_model


import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
# from transformers import Wav2Vec2Processor, 
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve

import torch
from torch.nn.utils.rnn import pad_sequence




class RawLabeledAudioDataset(Dataset):
    def __init__(self, directory,labels_dict, transform=None,normalize=True):
        """
        Args:
            directory (str): Path to the directory containing the audio files.
            labels_dict (dict): Dictionary of labels for each audio file.
            save_dir (str): Path to the directory where the extracted features will be saved.
            tokenizer (callable): A tokenizer for preprocessing the audio data.
            feature_extractor (callable): Feature extractor model (e.g., from HuggingFace).
            transform (callable, optional): Optional transform to apply to the waveform.
            normalize (bool, optional): Whether to normalize the waveform. Default is True.
        """
        self.directory = directory
        self.labels_dict = labels_dict
        # self.save_dir = save_dir
        # self.tokenizer = tokenizer
        # self.feature_extractor = feature_extractor
        self.transform = transform
        self.normalize = normalize
        self.file_list = [f for f in os.listdir(directory) if f.endswith('.wav')]

        # Ensure the save directory exists
        # os.makedirs(save_dir, exist_ok=True)

    def __len__(self):
        return len(self.file_list)


    def normalize_waveform(self, waveform):
        """
        Normalize the waveform by scaling it to [-1, 1] or applying Z-score normalization.
        
        Args:
            waveform (Tensor): The input waveform tensor.
        
        Returns:
            Tensor: The normalized waveform.
        """
        # Method 1: Normalize to [-1, 1]
        waveform = waveform / waveform.abs().max()

        # Method 2: Z-score normalization (mean=0, std=1)
        # waveform = (waveform - waveform.mean()) / waveform.std()

        return waveform
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.directory, file_name)

        try:
            # waveform, sample_rate = torchaudio.load(file_path, normalize=True)
            waveform, sample_rate = torchaudio.load(file_path, normalize=False)
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None
        
        # Normalize waveform if needed
        if self.normalize:
            waveform = self.normalize_waveform(waveform)

        # Apply any other transformations if provided
        if self.transform:
            waveform = self.transform(waveform)

        # Return raw waveform and sample rate
        file_name = file_name.split('.')[0]
        # label = self.labels_dict.get(file_name).astype(int)
        label = self.labels_dict.get(file_name)
        # label = torch.tensor(label, dtype=torch.int8)
        label = torch.tensor(label)

        return {'waveform': waveform, 'sample_rate': sample_rate, 'label': label, 'file_name': file_name}
    





def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Remove None values
    if len(batch) == 0:
        return None
    
    # waveforms = [item['waveform'] for item in batch]
    # labels = [item['label'] for item in batch]

    # Pad waveforms to have the same length
    # waveforms_padded=pad_sequence([waveform.squeeze(0) for waveform in waveforms], batch_first=True)
    waveforms_padded=pad_sequence([waveform.squeeze(0) for waveform in [item['waveform'] for item in batch]], batch_first=True)

    return {
        'waveform': waveforms_padded,
        'label': torch.stack([item['label'] for item in batch]),
        'sample_rate': [item['sample_rate'] for item in batch],
        'file_name': [item['file_name'] for item in batch]
    }










# from torch.utils.data import DataLoader
import torch.multiprocessing as mp

# Assuming AudioDataset and collate_fn are defined elsewhere
def get_raw_labeled_audio_data_loaders(directory, labels_dict, batch_size=32, shuffle=True, num_workers=0, prefetch_factor=None):
    
    # If multiprocessing is used, set start method to 'spawn' (for avoiding pickling issues)
    # if num_workers > 0:
    #     # mp.set_start_method('spawn', force=True)
    #     mp.set_start_method('fork', force=True)
    
    # Create the dataset instance
    dataset = RawLabeledAudioDataset(directory, labels_dict)
    
    # pitch_shift_transform = PitchShiftTransform(sample_rate=16000, pitch_shift_prob=0.5, pitch_shift_steps=(-2, 2))

    # # Initialize the dataset with the transform
    # dataset = RawLabeledAudioDataset(
    #     directory=directory,
    #     labels_dict=labels_dict,
    #     transform=pitch_shift_transform  # Apply pitch shift as part of the dataset transform
    # )


    # Create the DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=True,  # Enable page-locked memory for faster data transfer to GPU
        prefetch_factor=prefetch_factor,  # How many batches to prefetch per worker
        collate_fn=custom_collate_fn  # Custom collate function to handle variable-length inputs
    )
    
    return data_loader



def create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_loss):
    metrics_dict=dict()
    metrics_dict['utterance_eer']=utterance_eer
    metrics_dict['utterance_eer_threshold']=utterance_eer_threshold
    metrics_dict['epoch_loss']=epoch_loss
    return metrics_dict






def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        # Convert numpy array to a list
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        # Convert numpy float to a native Python float
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        # Convert numpy int to a native Python int
        return int(obj)
    elif isinstance(obj, dict):
        # Recursively convert dictionary values
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively convert list items
        return [convert_to_serializable(i) for i in obj]
    else:
        # Return the object if it is already serializable
        return obj


def save_json_dictionary(path,my_dict):
  import json

  try:
      with open(path, 'w') as json_file:
          # Convert dictionary to serializable format
          serializable_dict = convert_to_serializable(my_dict)
          json.dump(serializable_dict, json_file, indent=4)
      print(f"Dictionary saved to {path}")
  except PermissionError:
      print(f"Error: Permission denied to write to the file {path}.")
  except IOError as e:
      print(f"Error: {e}")

# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================


import torch
import torch.nn as nn
import torchaudio.models as tam
import math

# # binary classification model  max pooling after feature extractor
# class BinarySpoofingClassificationModel(nn.Module):
#     def __init__(self, feature_dim, num_heads, hidden_dim,max_dropout=0.2, depthwise_conv_kernel_size=31,conformer_layers=1):
#         super(BinarySpoofingClassificationModel, self).__init__()

#         # Max pooling layer before the Conformer block
#         self.max_pooling = nn.MaxPool1d(kernel_size=feature_dim // 256, stride=feature_dim // 256)  # Reduce feature dimension to 256
#         # self.max_pooling = nn.MaxPool1d(3, stride=3) # Reduce feature dimension to 256
        
#         self.max_dropout=max_dropout
#         # Define the Conformer model from torchaudio
#         self.conformer = tam.Conformer(
#             input_dim=256,
#             num_heads=num_heads,
#             ffn_dim=hidden_dim,  # Feed-forward network dimension (for consistency)
#             num_layers=conformer_layers,  # Example, adjust as needed
#             depthwise_conv_kernel_size=depthwise_conv_kernel_size,  # Set the kernel size for depthwise convolution
#             dropout=0.2,
#             use_group_norm= False, 
#             convolution_first= False
#         )
        
#         # Global pooling layer (SelfWeightedPooling)
#         self.pooling = SelfWeightedPooling(256, mean_only=True)  # Pool across sequence dimension
        
#         # Add a feedforward block for feature refinement before classification
#         self.fc_refinement = nn.Sequential(
#             nn.Linear(256, hidden_dim),  # Refined hidden dimension for classification
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim),
#             nn.Dropout(0.2),  # Dropout for regularization

#             nn.Linear(hidden_dim, hidden_dim//2),  # Refined hidden dimension for classification
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim//2),
#             nn.Dropout(0.2),  # Dropout for regularization

#             nn.Linear(hidden_dim//2, hidden_dim//4),  # Refined hidden dimension for classification
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim//4),
#             nn.Dropout(0.2),  # Dropout for regularization

#             nn.Linear(hidden_dim//4, 1),  # Final output layer
#             # nn.Sigmoid(),
#             # nn.GELU(),
#         )


#         self.apply(self.initialize_weights)

#     # Custom initialization for He and Xavier
#     def initialize_weights(self, m, bias_value=0.05):
#         if isinstance(m, nn.Linear):  # For Linear layers
#             # We do not directly check activation here, since it's separate
#             if isinstance(m, nn.Linear):
#                 if hasattr(m, 'activation') and isinstance(m.activation, nn.ReLU):
#                     # He (Kaiming) initialization for ReLU/GELU layers
#                     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 elif hasattr(m, 'activation') and isinstance(m.activation, nn.GELU):
#                     # He (Kaiming) initialization for GELU layers
#                     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 elif hasattr(m, 'activation') and isinstance(m.activation, (nn.Tanh, nn.Sigmoid)):
#                     # Xavier (Glorot) initialization for tanh/sigmoid layers
#                     nn.init.xavier_normal_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, bias_value)

#         elif isinstance(m, nn.Conv1d):  # For Conv1d layers (typically used in Conformer)
#             if hasattr(m, 'activation') and isinstance(m.activation, nn.ReLU):
#                 # He (Kaiming) initialization for Conv1d with ReLU/GELU
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif hasattr(m, 'activation') and isinstance(m.activation, (nn.Tanh, nn.Sigmoid)):
#                 # Xavier (Glorot) initialization for Conv1d with tanh/sigmoid
#                 nn.init.xavier_normal_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, bias_value)


#     def forward(self, x, lengths,dropout_prob):
#         # print(f" x size before conformer = {x.size()}")
        
#         x = self.max_pooling(x)  # Apply max pooling

#         # Apply Conformer model
#         x, _ = self.conformer(x, lengths)  # The second returned value is the sequence lengths
#         # print(f" x size after conformer = {x.size()}")
        
#         # Apply global pooling across the sequence dimension (SelfWeightedPooling)
#         x = self.pooling(x)  # Now x is (batch_size, hidden_dim, 1)
#         # print(f" x size after pooling = {x.size()}")

#         # Update the dropout probability dynamically
#         self.fc_refinement[3].p = dropout_prob  # Update the dropout layer's probability
#         self.fc_refinement[7].p = dropout_prob  # Update the dropout layer's probability
#         self.fc_refinement[11].p = dropout_prob  # Update the dropout layer's probability

#         # Refine features before classification using the fc_refinement block
#         utt_score = self.fc_refinement(x)
#         return utt_score # Return the classification output
#     def adjust_dropout(self, epoch, total_epochs):
#         # Cosine annealing for dropout probability
#         return self.max_dropout * (1 + math.cos(math.pi * epoch / total_epochs)) / 2


# binary classification model without max pooling after feature extractor
class BinarySpoofingClassificationModel(nn.Module):
    def __init__(self, feature_dim, num_heads, hidden_dim,max_dropout=0.2, depthwise_conv_kernel_size=31,conformer_layers=1):
        super(BinarySpoofingClassificationModel, self).__init__()

        # Max pooling layer before the Conformer block
        # self.max_pooling = nn.MaxPool1d(kernel_size=feature_dim // 256, stride=feature_dim // 256)  # Reduce feature dimension to 256
        # self.max_pooling = nn.MaxPool1d(3, stride=3) # Reduce feature dimension to 256
        
        self.max_dropout=max_dropout
        # Define the Conformer model from torchaudio
        self.conformer = tam.Conformer(
            input_dim=feature_dim,
            num_heads=num_heads,
            ffn_dim=hidden_dim,  # Feed-forward network dimension (for consistency)
            num_layers=conformer_layers,  # Example, adjust as needed
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,  # Set the kernel size for depthwise convolution
            dropout=0.2,
            use_group_norm= False, 
            convolution_first= False
        )
        
        # Global pooling layer (SelfWeightedPooling)
        self.pooling = SelfWeightedPooling(feature_dim, mean_only=True)  # Pool across sequence dimension
        
        # Add a feedforward block for feature refinement before classification
        self.fc_refinement = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),  # Refined hidden dimension for classification
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),  # Dropout for regularization

            nn.Linear(hidden_dim, hidden_dim//2),  # Refined hidden dimension for classification
            nn.GELU(),
            nn.LayerNorm(hidden_dim//2),
            nn.Dropout(0.2),  # Dropout for regularization

            nn.Linear(hidden_dim//2, hidden_dim//4),  # Refined hidden dimension for classification
            nn.GELU(),
            nn.LayerNorm(hidden_dim//4),
            nn.Dropout(0.2),  # Dropout for regularization

            nn.Linear(hidden_dim//4, 1),  # Final output layer
            # nn.Sigmoid(),
            # nn.GELU(),
        )


        self.apply(self.initialize_weights)

    # Custom initialization for He and Xavier
    def initialize_weights(self, m, bias_value=0.05):
        if isinstance(m, nn.Linear):  # For Linear layers
            # We do not directly check activation here, since it's separate
            if isinstance(m, nn.Linear):
                if hasattr(m, 'activation') and isinstance(m.activation, nn.ReLU):
                    # He (Kaiming) initialization for ReLU/GELU layers
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif hasattr(m, 'activation') and isinstance(m.activation, nn.GELU):
                    # He (Kaiming) initialization for GELU layers
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif hasattr(m, 'activation') and isinstance(m.activation, (nn.Tanh, nn.Sigmoid)):
                    # Xavier (Glorot) initialization for tanh/sigmoid layers
                    nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_value)

        elif isinstance(m, nn.Conv1d):  # For Conv1d layers (typically used in Conformer)
            if hasattr(m, 'activation') and isinstance(m.activation, nn.ReLU):
                # He (Kaiming) initialization for Conv1d with ReLU/GELU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif hasattr(m, 'activation') and isinstance(m.activation, (nn.Tanh, nn.Sigmoid)):
                # Xavier (Glorot) initialization for Conv1d with tanh/sigmoid
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_value)


    def forward(self, x, lengths,dropout_prob):
        # print(f" x size before conformer = {x.size()}")
        
        # x = self.max_pooling(x)  # Apply max pooling

        # Apply Conformer model
        x, _ = self.conformer(x, lengths)  # The second returned value is the sequence lengths
        # print(f" x size after conformer = {x.size()}")
        
        # Apply global pooling across the sequence dimension (SelfWeightedPooling)
        x = self.pooling(x)  # Now x is (batch_size, hidden_dim, 1)
        # print(f" x size after pooling = {x.size()}")

        # Update the dropout probability dynamically
        self.fc_refinement[3].p = dropout_prob  # Update the dropout layer's probability
        self.fc_refinement[7].p = dropout_prob  # Update the dropout layer's probability
        self.fc_refinement[11].p = dropout_prob  # Update the dropout layer's probability

        # Refine features before classification using the fc_refinement block
        utt_score = self.fc_refinement(x)
        return utt_score # Return the classification output
    def adjust_dropout(self, epoch, total_epochs):
        # Cosine annealing for dropout probability
        return self.max_dropout * (1 + math.cos(math.pi * epoch / total_epochs)) / 2
        




def dev_model( PS_Model,dev_directory, labels_dict,feature_extractor,dropout_prob, BATCH_SIZE=32,epoch=0,DEVICE='cpu'):

    BASE_DIR = os.getcwd()
    # Get the data loader

    # dev_loader = get_audio_data_loaders(dev_directory, labels_dict, tokenizer,feature_extractor, batch_size=BATCH_SIZE, shuffle=True)
    # dev_loader = get_raw_labeled_audio_data_loaders(dev_directory, labels_dict,batch_size=BATCH_SIZE, shuffle=True, num_workers=8, prefetch_factor=2)
    dev_loader = get_raw_labeled_audio_data_loaders(dev_directory, labels_dict,batch_size=BATCH_SIZE, shuffle=True)
    
    # Validation phase
    PS_Model.eval()  # Set the model to evaluation mode

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        PS_Model = nn.DataParallel(PS_Model).to(DEVICE)
        print("Parallelizing model on ", torch.cuda.device_count(), "GPUs!")



    # Define loss criterion
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)

    files_names=[]

    epoch_loss = 0
    utterance_eer, utterance_eer_threshold=0,0
    utterance_predictions=[]
    # c=0
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Dev Batches", leave=False):
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
            outputs = PS_Model(features,lengths,dropout_prob)

            # Calculate loss
            loss = criterion(outputs, labels) 
            if torch.isnan(loss).any(): 
                print(f"NaN detected in loss at epoch {epoch}") 
                continue
            epoch_loss += loss.item()

            with torch.no_grad():
                # Calculate utterance predictions
                utterance_predictions.extend(outputs)
                # Accumulate files names
                files_names.extend(batch['file_name'])


        # Get Average Utterance EER for the epoch
        utterance_labels =[labels_dict[file_name] for file_name in files_names]
        # print(f'epoch {epoch} , utterance_labels: {utterance_labels}')
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_eer(utterance_predictions,torch.tensor(utterance_labels))

        # Average loss for the epoch
        epoch_loss /= len(dev_loader)


    # Print epoch dev progress
    # print(f'Epoch [{epoch + 1}] Complete. Validation Loss: {epoch_loss:.4f},\n'
    #            f'Average Validation Segment EER: {segment_eer:.4f}, Average Validation Segment EER Threshold: {segment_eer_threshold:.4f},\n'
    #            f'Average Validation Utterance EER: {utterance_eer:.4f}, Average Validation Utterance EER Threshold: {utterance_eer_threshold:.4f}')

    return create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_loss)
    





def infer_model(model_path,test_directory, test_labels_dict,feature_extractor, BATCH_SIZE=32,DEVICE='cpu'):
    # Initialize the model
    hidd_dims ={'wav2vec2':768, 'wav2vec2_large':1024}
    PS_Model = BinarySpoofingClassificationModel(feature_dim=hidd_dims['wav2vec2'], num_heads=8, hidden_dim=128,conformer_layers=1).to(DEVICE)  # Move model to the configured device
    # PS_Model,_,_=load_checkpoint(PS_Model, optimizer, path=os.path.join(os.getcwd(),'models/back_end_models/model_epochs30_batch8_lr0.005_20241216_013405.pth'))
    checkpoint = torch.load(model_path)
    PS_Model.load_state_dict(checkpoint['model_state_dict'])

    PS_Model.eval()  # Set the model to evaluation mode

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        PS_Model = nn.DataParallel(PS_Model).to(DEVICE)
        print("Parallelizing model on ", torch.cuda.device_count(), "GPUs!")


    # Get the test data loader
    # test_loader = get_audio_data_loaders(test_directory, test_labels_dict, tokenizer, feature_extractor, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = get_raw_labeled_audio_data_loaders(test_directory, test_labels_dict,batch_size=BATCH_SIZE, shuffle=True, num_workers=8, prefetch_factor=2)

    # Get Utterance lables dictionary    
    BASE_DIR = os.getcwd()
    # PartialSpoof_LA_cm_eval_trl_dict_path = os.path.join(BASE_DIR,'database/utterance_labels/PartialSpoof_LA_cm_eval_trl.json')
    # PartialSpoof_LA_cm_eval_trl_dict= load_json_dictionary(PartialSpoof_LA_cm_eval_trl_dict_path)

    # Define loss criterion
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)

    files_names=[]

    epoch_loss = 0
    utterance_eer, utterance_eer_threshold=0,0
    utterance_predictions=[]

    dropout_prob=0

    with torch.no_grad():  # Disable gradient calculation during inference
        for batch in tqdm(test_loader, desc="Test Batches", leave=False):
            waveforms = batch['waveform'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            labels = labels.unsqueeze(1).float()   # Converts labels from shape [batch_size] to [batch_size, 1]

            # Forward pass through wav2vec2 for feature extraction
            features = feature_extractor(waveforms)['hidden_states'][-1] 

            # lengths should be the number of non-padded frames in each sequence
            lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16).to(DEVICE)  # (batch_size,)

            # Pass features to model and get predictions
            outputs = PS_Model(features,lengths,dropout_prob)

            loss = criterion(outputs, labels) 
            epoch_loss += loss.item()


            with torch.no_grad():
                # Calculate utterance predictions
                utterance_predictions.extend(outputs)
                # Accumulate files names
                files_names.extend(batch['file_name'])


        # Get Average Utterance EER for the epoch
        # utterance_labels =[PartialSpoof_LA_cm_eval_trl_dict[file_name] for file_name in files_names]
        utterance_labels =[test_labels_dict[file_name] for file_name in files_names]        
        # print(f'epoch {epoch} , utterance_labels: {utterance_labels}')
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_eer(utterance_predictions,torch.tensor(utterance_labels))

        # Average loss for the epoch
        epoch_loss /= len(test_loader)


    # Print epoch dev progress
    print(f'Testing/Inference Complete. Test Loss: {epoch_loss:.4f},\n'
               f'Average Test Utterance EER: {utterance_eer:.4f}, Average Test Utterance EER Threshold: {utterance_eer_threshold:.4f}')

    return create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_loss)






def train_model(train_directory, train_labels_dict, 
                BATCH_SIZE=32, NUM_EPOCHS=1,LEARNING_RATE=0.0001,
                model_save_path=os.path.join(os.getcwd(),'models/back_end_models'),
                DEVICE='cpu',save_interval=float('inf'),patience=10,save_feature_extractor=False,max_grad_norm=1.0,monitor_dev_epoch=0):

    # Initialize W&B
    wandb.init(project='partial_spoof_Wav2Vec2_Conformer_binary_classifier')

    # # Initialize early stopping
    # early_stopping = EarlyStopping(patience=patience, verbose=True)

    if DEVICE == 'cuda':torch.cuda.empty_cache()
    # Ensure the model save path exists
    os.makedirs(model_save_path, exist_ok=True)
    # Load utterance labels
    BASE_DIR = os.getcwd()
    # PartialSpoof_LA_cm_train_trl_dict_path = os.path.join(BASE_DIR,'database/utterance_labels/PartialSpoof_LA_cm_train_trl.json')
    # PartialSpoof_LA_cm_train_trl_dict= load_json_dictionary(PartialSpoof_LA_cm_train_trl_dict_path)

    # Load feature extractor
    ssl_ckpt_path = os.path.join(os.getcwd(), 'models/w2v_large_lv_fsh_swbd_cv.pt')
    feature_extractor = torch.hub.load('s3prl/s3prl', 'wav2vec2', model_path=ssl_ckpt_path).to(DEVICE)

    # Initialize the model, loss function, and optimizer
    hidd_dims ={'wav2vec2':768, 'wav2vec2_large':1024}
    PS_Model = BinarySpoofingClassificationModel(feature_dim=hidd_dims['wav2vec2'], num_heads=8, hidden_dim=128,conformer_layers=1).to(DEVICE)  # Move model to the configured device

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
    # criterion = CustomLoss().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss().to(DEVICE)


    gamma=0.9
    LR_SCHEDULER = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # Get the data loader
    # train_loader = get_raw_labeled_audio_data_loaders(train_directory, train_labels_dict,batch_size=BATCH_SIZE, shuffle=True, num_workers=8, prefetch_factor=2)
    train_loader = get_raw_labeled_audio_data_loaders(train_directory, train_labels_dict,batch_size=BATCH_SIZE, shuffle=True)


    # PS_Model,optimizer,_=load_checkpoint(PS_Model, optimizer, path=os.path.join(os.getcwd(),'models/back_end_models/model_epochs1_batch16_lr0.0002355064348623125_20241217_000934.pth'))
    # PS_Model,optimizer,_=load_checkpoint(PS_Model, optimizer, path=os.path.join(os.getcwd(),'models/back_end_models/model_epochs1_batch8_lr0.00021195579137608128_20241218_154600.pth'))

    # Logging gradients with wandb.watch
    wandb.watch(PS_Model, log_freq=100,log='all')

    PS_Model.train()  # Set the model to training mode

    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
        PS_Model.train()  # Set the model to training mode

        # Adjust dropout probability for the current epoch
        dropout_prob = PS_Model.adjust_dropout(epoch, NUM_EPOCHS)

        epoch_loss = 0
        utterance_eer, utterance_eer_threshold=0,0
        utterance_predictions=[]
        files_names=[]
        # c=0
        for batch in tqdm(train_loader, desc="Train Batches", leave=False):
            # if c>8:
            #     break
            # else:
            #     c+=1
            waveforms = batch['waveform'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            labels = labels.unsqueeze(1).float()   # Converts labels from shape [batch_size] to [batch_size, 1]
            # print(f"labels : {labels} , type(labels) : {type(labels)}")

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
            # print(f"PS_Model outputs: {outputs} , type(outputs) : {type(outputs)} ")

            # Calculate loss
            loss = criterion(outputs, labels)  
            if torch.isnan(loss).any(): 
                print(f"NaN detected in loss at epoch {epoch}") 
                continue
            epoch_loss += loss.item()


            # Backward pass and optimization
            loss.backward()
            # Apply gradient clipping to prevent vanishing/exploding gradients
            torch.nn.utils.clip_grad_norm_(PS_Model.parameters(), max_grad_norm)
    
            optimizer.step()


            with torch.no_grad():  # No need to compute gradients for EER calculation
                # Calculate utterance predictions
                utterance_predictions.extend(outputs)
                # Accumulate files names
                files_names.extend(batch['file_name'])


        # if DEVICE=='cuda': torch.cuda.empty_cache()

        # Save checkpoint
        if NUM_EPOCHS>=save_interval and (epoch + 1) % (save_interval) == 0:
            # Generate a unique filename based on hyperparameters
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"model_epochs{epoch + 1}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pth"
                        
            save_checkpoint(PS_Model, optimizer, epoch + 1,os.path.join(model_save_path,model_filename))


        # Get Average Utterance EER for the epoch
        utterance_labels =[train_labels_dict[file_name] for file_name in files_names]
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_eer(utterance_predictions,torch.tensor(utterance_labels))
        # utterance_pooling_predictions = torch.cat(utterance_pooling_predictions, dim=0)
        # utterance_pooling_eer, utterance_pooling_eer_threshold = compute_eer(utterance_pooling_predictions,torch.tensor(utterance_labels))

        # Average  loss for the epoch
        epoch_loss /= len(train_loader)



        if (epoch+1) >= monitor_dev_epoch :
            # print("dev_model not ready yet ! ")
            BASE_DIR = os.getcwd()
            # Define development files and labels
            dev_files_path=os.path.join(BASE_DIR,'database/dev/con_wav')
            # dev_seglab_64_path=os.path.join(BASE_DIR,'database/segment_labels/dev_seglab_0.64.npy')
            # dev_seglab_64_dict = np.load(dev_seglab_64_path, allow_pickle=True).item()
            PartialSpoof_LA_cm_dev_trl_dict_path = os.path.join(BASE_DIR,'database/utterance_labels/PartialSpoof_LA_cm_dev_trl.json')
            dev_seglab_64_dict= load_json_dictionary(PartialSpoof_LA_cm_dev_trl_dict_path)

            dev_metrics_dict=dev_model( PS_Model,dev_files_path, dev_seglab_64_dict,feature_extractor,dropout_prob, BATCH_SIZE,DEVICE=DEVICE)

            wandb.log({'epoch': epoch+1,'training_loss_epoch': epoch_loss,
                'training_utterance_eer_epoch': utterance_eer,
                'training_utterance_eer_threshold_epoch': utterance_eer_threshold, 
                # 'training_utterance_pooling_eer_epoch': utterance_pooling_eer,
                # 'training_utterance_pooling_eer_threshold_epoch': utterance_pooling_eer_threshold, 
                'validation_loss_epoch': dev_metrics_dict['epoch_loss'],
                'validation_utterance_eer_epoch': dev_metrics_dict['utterance_eer'],
                'validation_utterance_eer_threshold_epoch': dev_metrics_dict['utterance_eer_threshold']                      
                })
        else:
            wandb.log({'epoch': epoch+1,'training_loss_epoch': epoch_loss,
                'training_utterance_eer_epoch': utterance_eer,
                'training_utterance_eer_threshold_epoch': utterance_eer_threshold                  
                })

        # Early stopping check
        # early_stopping(dev_metrics_dict['epoch_loss'], PS_Model)
        # if early_stopping.early_stop:
        #     print(f"Early stopping at epoch {epoch+1}")
        #     break

        LR_SCHEDULER.step()


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
    torch.save(utterance_predictions,os.path.join(os.getcwd(),f'outputs/utterance_predictions_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pt'))
    torch.save(torch.tensor(utterance_labels),os.path.join(os.getcwd(),f'outputs/utterance_labels_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pt'))

    # Save last metrics
    training_metrics_dict=create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_loss)
    training_metrics_dict_filename = f"metrics_dict_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.json"
    training_metrics_dict_save_path=os.path.join(os.getcwd(),f'outputs/{training_metrics_dict_filename}')
    save_json_dictionary(training_metrics_dict_save_path,training_metrics_dict)

    if DEVICE=='cuda': torch.cuda.empty_cache()
    wandb.finish()
    print("Training complete!")



def train():
    # Initialize W&B
    wandb.init(project='partial_spoof_Wav2Vec2_Conformer_binary_classifier')

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
    # train_seglab_64_path=os.path.join(BASE_DIR,'database/segment_labels/train_seglab_0.64.npy')
    # train_seglab_64_dict = np.load(train_seglab_64_path, allow_pickle=True).item()
    PartialSpoof_LA_cm_train_trl_dict_path = os.path.join(BASE_DIR,'database/utterance_labels/PartialSpoof_LA_cm_train_trl.json')
    PartialSpoof_LA_cm_train_trl_dict= load_json_dictionary(PartialSpoof_LA_cm_train_trl_dict_path)


    # Call train_model with parameters from W&B sweep
    train_model(
        train_directory=train_files_path,
        train_labels_dict=PartialSpoof_LA_cm_train_trl_dict,
        BATCH_SIZE=config.BATCH_SIZE,
        NUM_EPOCHS=config.NUM_EPOCHS,
        LEARNING_RATE=config.LEARNING_RATE,
        DEVICE=DEVICE,
        save_interval=10
    )



def inference():
    print("infer_model is working ... ")
    # Get Device
    use_cuda= True
    use_cuda =  use_cuda and torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    print(f'device: {DEVICE}')

    # Define your paths and other fixed arguments
    BASE_DIR = os.getcwd()

    # Define training files and labels
    eval_files_path=os.path.join(BASE_DIR,'database/eval/con_wav')

    PartialSpoof_LA_cm_eval_trl_dict_path = os.path.join(BASE_DIR,'database/utterance_labels/PartialSpoof_LA_cm_eval_trl.json')
    test_labels_dict= load_json_dictionary(PartialSpoof_LA_cm_eval_trl_dict_path)


    # Load feature extractor
    ssl_ckpt_path = os.path.join(os.getcwd(), 'models/w2v_large_lv_fsh_swbd_cv.pt')
    feature_extractor = torch.hub.load('s3prl/s3prl', 'wav2vec2', model_path=ssl_ckpt_path).to(DEVICE)
    feature_extractor.eval()

    model_path=os.path.join(os.getcwd(),'models/back_end_models/model_epochs1_batch16_lr0.0002355064348623125_20241217_000934.pth')

    BATCH_SIZE=16
    inference_metrics_dict=infer_model(
        model_path=model_path,
        test_directory=eval_files_path,
        test_labels_dict=test_labels_dict,
        feature_extractor=feature_extractor, 
        BATCH_SIZE=BATCH_SIZE,
        DEVICE=DEVICE
    )

    if DEVICE=='cuda': torch.cuda.empty_cache()




import wandb
import os
import random
from datetime import datetime


def main():
    """ main(): the default wrapper for training and inference process
    """

    # wandb_key="Get the key here"
    wandb_api_key="c1fc533d0bafe63c83a9110c6daef36b76f77de1"
    # os.system(f"echo {wandb_key}")
    wandb.login(key=wandb_api_key,relogin=True,force=True)

    # wandb.init(project='partial_spoof_demo')


    sweep_config = {
        'method': 'bayes',
        'metric': 
        {
            'goal': 'minimize', 
            'name': 'validation_utterance_eer_epoch'
            },
        'parameters': 
        {
            # 'NUM_EPOCHS': {'values': [5, 7]},
            # 'LEARNING_RATE': {'values': [0.001]},
            # 'BATCH_SIZE': {'values': [16,32]},
            'NUM_EPOCHS': {'values': [30]},
            'LEARNING_RATE': {'values': [0.005]},
            # 'LEARNING_RATE': {'values': [0.00021195579137608126]},
            'BATCH_SIZE': {'values': [8]},
            # 'CLASS0_WEIGHT': {'values': [0.42,0.45,0.48]},

        }
    }

    sweep_id = wandb.sweep(sweep=sweep_config,project='partial_spoof_Wav2Vec2_Conformer_binary_classifier')
    # sweep_id = wandb.sweep(sweep=sweep_config)
    wandb.agent(sweep_id, function=train, count=1)




if __name__ == "__main__":
    # Record the start time
    start_time = datetime.now()

    main()
    # inference()

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


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

import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader , ConcatDataset
# from transformers import Wav2Vec2Processor, 
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve

import torch
from torch.nn.utils.rnn import pad_sequence

# from torch.utils.data import DataLoader
import torch.multiprocessing as mp


import torch
import torch.nn as torch_nn
import torchaudio
import torch.nn.functional as torch_nn_func

# ============================================================================================
# SAP = SelfWeightedPooling

import torch.nn.init as torch_init


class SelfWeightedPooling(torch_nn.Module):
    """ SelfWeightedPooling module
    Inspired by
    https://github.com/joaomonteirof/e2e_antispoofing/blob/master/model.py
    To avoid confusion, I will call it self weighted pooling
    
    l_selfpool = SelfWeightedPooling(5, 1, False)
    with torch.no_grad():
        input_data = torch.rand([3, 10, 5])
        output_data = l_selfpool(input_data)
    """
    def __init__(self, feature_dim, num_head=1, mean_only=False):
        """ SelfWeightedPooling(feature_dim, num_head=1, mean_only=False)
        Attention-based pooling
        
        input (batchsize, length, feature_dim) ->
        output 
           (batchsize, feature_dim * num_head), when mean_only=True
           (batchsize, feature_dim * num_head * 2), when mean_only=False
        
        args
        ----
          feature_dim: dimension of input tensor
          num_head: number of heads of attention
          mean_only: whether compute mean or mean with std
                     False: output will be (batchsize, feature_dim*2)
                     True: output will be (batchsize, feature_dim)
        """
        super(SelfWeightedPooling, self).__init__()

        self.feature_dim = feature_dim
        self.mean_only = mean_only
        self.noise_std = 1e-5
        self.num_head = num_head

        # transformation matrix (num_head, feature_dim)
        self.mm_weights = torch_nn.Parameter(
            torch.Tensor(num_head, feature_dim), requires_grad=True)
        torch_init.kaiming_uniform_(self.mm_weights)
        return
    
    def _forward(self, inputs):
        """ output, attention = forward(inputs)
        inputs
        ------
          inputs: tensor, shape (batchsize, length, feature_dim)
        
        output
        ------
          output: tensor
           (batchsize, feature_dim * num_head), when mean_only=True
           (batchsize, feature_dim * num_head * 2), when mean_only=False
          attention: tensor, shape (batchsize, length, num_head)
        """        
        # batch size
        batch_size = inputs.size(0)
        # feature dimension
        feat_dim = inputs.size(2)
        
        # input is (batch, legth, feature_dim)
        # change mm_weights to (batchsize, feature_dim, num_head)
        # weights will be in shape (batchsize, length, num_head)
        weights = torch.bmm(inputs, 
                            self.mm_weights.permute(1, 0).contiguous()\
                            .unsqueeze(0).repeat(batch_size, 1, 1))
        
        # attention (batchsize, length, num_head)
        attentions = torch_nn_func.softmax(torch.tanh(weights),dim=1)        
        
        # apply attention weight to input vectors
        if self.num_head == 1:
            # We can use the mode below to compute self.num_head too
            # But there is numerical difference.
            #  original implementation in github
            
            # elmentwise multiplication
            # weighted input vector: (batchsize, length, feature_dim)
            weighted = torch.mul(inputs, attentions.expand_as(inputs))
        else:
            # weights_mat = (batch * length, feat_dim, num_head)
            #    inputs.view(-1, feat_dim, 1), zl, error
            #    RuntimeError: view size is not compatible with input tensor's size and stride 
            #    (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
            weighted = torch.bmm(
                inputs.reshape(-1, feat_dim, 1), 
                attentions.view(-1, 1, self.num_head))
            
            # weights_mat = (batch, length, feat_dim * num_head)
            weighted = weighted.view(batch_size, -1, feat_dim * self.num_head)
            
        # pooling
        if self.mean_only:
            # only output the mean vector
            representations = weighted.sum(1)
        else:
            # output the mean and std vector
            noise = self.noise_std * torch.randn(
                weighted.size(), dtype=weighted.dtype, device=weighted.device)

            avg_repr, std_repr = weighted.sum(1), (weighted+noise).std(1)
            # concatenate mean and std
            representations = torch.cat((avg_repr,std_repr),1)
        # done
        return representations, attentions
    
    def forward(self, inputs):
        """ output = forward(inputs)
        inputs
        ------
          inputs: tensor, shape (batchsize, length, feature_dim)
        
        output
        ------
          output: tensor
           (batchsize, feature_dim * num_head), when mean_only=True
           (batchsize, feature_dim * num_head * 2), when mean_only=False
        """
        output, _ = self._forward(inputs)
        return output

    def debug(self, inputs):
        return self._forward(inputs)
# ============================================================================================

import random
import torchaudio.transforms as T
from torch.utils.data import Dataset
import torch

class PitchShiftTransform:
    def __init__(self, sample_rate, pitch_shift_prob=0.5, pitch_shift_steps=(-3, 3)):
        """
        Args:
            sample_rate (int): The sample rate of the audio data.
            pitch_shift_prob (float): Probability of applying pitch shift augmentation.
            pitch_shift_steps (tuple): Range of pitch shift steps in semitones (e.g., (-4, 4)).
        """
        self.sample_rate = sample_rate
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_steps = pitch_shift_steps

    def __call__(self, waveform):
        """
        Apply pitch shift with the given probability.
        
        Args:
            waveform (Tensor): The input audio waveform tensor.
        
        Returns:
            Tensor: The pitch-shifted waveform if the probability condition is met, otherwise original waveform.
        """
        # if random.random() < self.pitch_shift_prob:
        if self.pitch_shift_prob > 0:
            n_steps = random.randint(*self.pitch_shift_steps)
            pitch_shift = T.PitchShift(self.sample_rate, n_steps)
            # waveform=waveform.detach()
            waveform = pitch_shift(waveform)
            # print(f"applied pitch_shifted with {n_steps} steps")
        return waveform



# ============================================================================================

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

    # Pad waveforms to have the same length
    waveforms_padded=pad_sequence([waveform.squeeze(0) for waveform in [item['waveform'] for item in batch]], batch_first=True)

    return {
        'waveform': waveforms_padded,
        'label': torch.stack([item['label'] for item in batch]),
        'sample_rate': [item['sample_rate'] for item in batch],
        'file_name': [item['file_name'] for item in batch]
    }









# binary classification model  max pooling after feature extractor
class BinarySpoofingClassificationModel(nn.Module):
    def __init__(self, feature_dim, num_heads, hidden_dim, max_dropout=0.2, depthwise_conv_kernel_size=31, conformer_layers=1, max_pooling_factor=3):
        super(BinarySpoofingClassificationModel, self).__init__()

        self.max_pooling_factor = max_pooling_factor
        self.feature_dim = feature_dim
        self.max_dropout=max_dropout

        if self.max_pooling_factor is not None:
            self.max_pooling = nn.MaxPool1d(kernel_size=self.max_pooling_factor, stride=self.max_pooling_factor)
            self.feature_dim=feature_dim//self.max_pooling_factor
        else:
            self.max_pooling = None
        
        print(f"self.feature_dim= {self.feature_dim} , self.max_pooling= {self.max_pooling}")
        # Define the Conformer model from torchaudio
        self.conformer = tam.Conformer(
            input_dim=self.feature_dim,
            num_heads=num_heads,
            ffn_dim=hidden_dim,  # Feed-forward network dimension (for consistency)
            num_layers=conformer_layers,  # Example, adjust as needed
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,  # Set the kernel size for depthwise convolution
            dropout=0.2,
            use_group_norm= False, 
            convolution_first= False
        )
        
        # Global pooling layer (SelfWeightedPooling)
        self.pooling = SelfWeightedPooling(self.feature_dim , mean_only=True)  # Pool across sequence dimension
        
        # Add a feedforward block for feature refinement before classification
        self.fc_refinement = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),  # Refined hidden dimension for classification
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
        if self.max_pooling is not None:
            x = self.max_pooling(x)  # Apply max pooling

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





# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================

def create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_loss):
    metrics_dict=dict()
    metrics_dict['utterance_eer']=utterance_eer
    metrics_dict['utterance_eer_threshold']=utterance_eer_threshold
    metrics_dict['epoch_loss']=epoch_loss
    return metrics_dict


def load_json_dictionary(path):
  import json

  # Define the path to your JSON file
  # input_file_path = os.path.join(BASE_DIR,'PartialSpoof_LA_cm_eval_trl.json')

  # Load the dictionary from the JSON file
  with open(path, 'r') as json_file:
      my_dict = json.load(json_file)

  return my_dict



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


def save_checkpoint(model, optimizer, epoch, path='checkpoint.pth'):
    # os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

import os
import torch

def load_checkpoint(model, optimizer, path='checkpoint.pth'):
    # Check if the file exists
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    # Load the checkpoint
    checkpoint = torch.load(path)

    # Verify the checkpoint contains the necessary keys
    if 'model_state_dict' not in checkpoint or 'optimizer_state_dict' not in checkpoint or 'epoch' not in checkpoint:
        raise KeyError(f"Checkpoint file is missing required keys ('model_state_dict', 'optimizer_state_dict', 'epoch')")

    # Check if the model state_dict is not empty
    if not checkpoint['model_state_dict']:
        raise ValueError(f"Model state_dict is empty in the checkpoint file at {path}")

    # Load the model and optimizer state dicts
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Return the model, optimizer, and epoch number
    return model, optimizer, checkpoint['epoch']


def compute_eer(predictions, labels):

    # Mask padding value
    # predictions, labels =get_masked_labels_and_outputs(predictions, labels)
    # print(f"after Mask padding value,\n nontarget_scores=\n{nontarget_scores} target_scores=\n{target_scores} ")
    # print(f"after Masking,\n predictions= {predictions} \n labels= {labels}")
    # Ensure scores and labels are PyTorch tensors and detach them
    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    if labels.ndim > 1 and labels.shape[0] == predictions.shape[0]:
        raise ValueError("labels dimension > 1, 1D vector is only supported for EER computation")
    else:
        # Compute false positive rate (fpr), true positive rate (tpr), and thresholds
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        
        # False Rejection Rate (FRR) is equal to 1 - TPR
        fnr = 1 - tpr

        # Check for NaN values
        if np.any(np.isnan(fnr)) or np.any(np.isnan(fpr)):
            raise ValueError("NaN values found in fnr or fpr. Cannot compute EER.")

        # Find the threshold where fpr (FAR) and frr are closest
        eer_threshold_index = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2  # EER is the point where FAR ≈ FRR
        
        # EER value and threshold where it occurs
        eer_threshold = thresholds[eer_threshold_index]
        
        return eer, eer_threshold


# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================
# Modularized helper functions

def initialize_wandb():
    """Initialize Weights & Biases for logging"""
    # wandb.init(project='partial_spoof_Wav2Vec2_Conformer_binary_classifier')
    wandb.init()

def initialize_models(ssl_ckpt_path, save_feature_extractor=False,
                      feature_dim=768, num_heads=8, hidden_dim=128, max_dropout=0.2, depthwise_conv_kernel_size=31, conformer_layers=1, max_pooling_factor=3, 
                      LEARNING_RATE=0.0001,DEVICE='cpu'):
    """Initialize the model, feature extractor, and optimizer"""
    # Initialize feature extractor
    if os.path.exists(ssl_ckpt_path):
        feature_extractor = torch.hub.load('s3prl/s3prl', 'wav2vec2', model_path=ssl_ckpt_path).to(DEVICE)
    else:
        ssl_ckpt_path = os.path.join(os.getcwd(), 'models/w2v_large_lv_fsh_swbd_cv.pt')
        feature_extractor = torch.hub.load('s3prl/s3prl', 'wav2vec2', model_path=ssl_ckpt_path).to(DEVICE)

    # Initialize Binary Spoofing Classification Model
    PS_Model = BinarySpoofingClassificationModel(feature_dim, num_heads, hidden_dim, max_dropout, depthwise_conv_kernel_size, conformer_layers, max_pooling_factor).to(DEVICE)

    # Freeze feature extractor if necessary
    if save_feature_extractor:
        for name, param in feature_extractor.named_parameters():
            if 'final_proj' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    # Optimizer setup
    optimizer = optim.AdamW(
        [{'params': feature_extractor.parameters(), 'lr': LEARNING_RATE / 10},
         {'params': PS_Model.parameters()}], 
        lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8) if save_feature_extractor else optim.AdamW(PS_Model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)

    return PS_Model, feature_extractor, optimizer

def initialize_data_loader(data_path, labels_path,BATCH_SIZE=32, shuffle=True, num_workers=0, prefetch_factor=None,pin_memory=False,apply_transform=False):
    """Initialize and return the training data loader"""
    labels_dict= load_json_dictionary(labels_path)

        # If multiprocessing is used, set start method to 'spawn' (for avoiding pickling issues)
    if num_workers > 0:
        if os.name == 'nt':  # Windows
            mp.set_start_method('spawn', force=True)
        else:  # Unix-based (Linux, macOS, etc.)
            mp.set_start_method('fork', force=True)
    
    # Create the dataset instance
    combined_dataset = RawLabeledAudioDataset(data_path, labels_dict)
    
    if apply_transform:
        # Apply pitch shift transform
        pitch_shift_transform = PitchShiftTransform(sample_rate=16000, pitch_shift_prob=1.0, pitch_shift_steps=(-2, 2))

        # # Initialize the dataset with the transform
        augmented_dataset = RawLabeledAudioDataset(
            directory=data_path,
            labels_dict=labels_dict,
            transform=pitch_shift_transform  # Apply pitch shift as part of the dataset transform
        )

        # Combine datasets
        combined_dataset = ConcatDataset([combined_dataset, augmented_dataset])

    # Create the DataLoader
    return DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory,  # Enable page-locked memory for faster data transfer to GPU
        prefetch_factor=prefetch_factor,  # How many batches to prefetch per worker
        collate_fn=custom_collate_fn  # Custom collate function to handle variable-length inputs
    )
    

def initialize_loss_function():
    """Initialize the loss function (BCE with logits)"""
    return nn.BCEWithLogitsLoss()

def adjust_dropout_prob(model, epoch, NUM_EPOCHS):
    """Adjust dropout rate dynamically during training"""
    return model.adjust_dropout(epoch, NUM_EPOCHS)

def save_checkpoint(model, optimizer, epoch, save_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)

def compute_metrics(outputs, labels):
    """Compute and return EER and other metrics"""
    utterance_eer, utterance_eer_threshold = compute_eer(outputs, labels)
    return utterance_eer, utterance_eer_threshold

def forward_pass(model, features, lengths, dropout_prob):
    """Forward pass through the model"""
    return model(features, lengths, dropout_prob)

def backward_and_optimize(model, loss, optimizer, max_grad_norm):
    """Backward pass and optimizer step"""
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

def log_metrics_to_wandb(epoch, epoch_loss, utterance_eer, utterance_eer_threshold, dev_metrics_dict=None):
    """Log metrics to W&B"""
    if dev_metrics_dict:
        wandb.log({
            'epoch': epoch + 1,
            'training_loss_epoch': epoch_loss,
            'training_utterance_eer_epoch': utterance_eer,
            'training_utterance_eer_threshold_epoch': utterance_eer_threshold,
            'validation_loss_epoch': dev_metrics_dict['epoch_loss'],
            'validation_utterance_eer_epoch': dev_metrics_dict['utterance_eer'],
            'validation_utterance_eer_threshold_epoch': dev_metrics_dict['utterance_eer_threshold'],
        })
    else:
        wandb.log({
            'epoch': epoch + 1,
            'training_loss_epoch': epoch_loss,
            'training_utterance_eer_epoch': utterance_eer,
            'training_utterance_eer_threshold_epoch': utterance_eer_threshold,
        })
    

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





def inference(DEVICE='cpu'):
    print("infer_model is working ... ")
    # Get Device
    use_cuda= True
    use_cuda =  use_cuda and torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    print(f'inference device: {DEVICE}')

    # Define your paths and other fixed arguments
    BASE_DIR = os.getcwd()

    # Define training files and labels
    eval_data_path=os.path.join(BASE_DIR,'database/eval/con_wav')
    eval_labels_path = os.path.join(BASE_DIR,'database/utterance_labels/PartialSpoof_LA_cm_eval_trl.json')
    eval_labels_dict= load_json_dictionary(eval_labels_path)
    BATCH_SIZE=16
    num_workers=8 
    prefetch_factor=2
    pin_memory= True if DEVICE=='cuda' else False   # Enable page-locked memory for faster data transfer to GPU
    eval_data_loader = initialize_data_loader(eval_data_path, eval_labels_path,BATCH_SIZE,False, num_workers, prefetch_factor,pin_memory)


    # Load feature extractor
    # ssl_ckpt_path = os.path.join(os.getcwd(), 'models/w2v_large_lv_fsh_swbd_cv.pt')
    ssl_ckpt_name='w2v_large_lv_fsh_swbd_cv_20241223_152156.pt'
    ssl_ckpt_path = os.path.join(os.getcwd(), f'models/back_end_models/{ssl_ckpt_name}')
    feature_extractor = torch.hub.load('s3prl/s3prl', 'wav2vec2', model_path=ssl_ckpt_path).to(DEVICE)
    feature_extractor.eval()

    # Initialize Binary Spoofing Classification Model
    PS_Model_name='model_epochs30_batch8_lr0.005_20241223_152156.pth'
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


    criterion = initialize_loss_function()


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

# ===========================================================================================================================
def train_one_epoch(model, train_loader, feature_extractor, optimizer, criterion, max_grad_norm, dropout_prob=0, DEVICE='cpu'):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    # utterance_eer, utterance_eer_threshold = 0, 0
    utterance_predictions = []
    files_names = []

    for batch in tqdm(train_loader, desc="Train Batches", leave=False):
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
    # early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Initialize model, feature extractor, optimizer, loss function
    PS_Model, feature_extractor, optimizer = initialize_models(ssl_ckpt_path, save_feature_extractor,
                      feature_dim, num_heads,hidden_dim,max_dropout,depthwise_conv_kernel_size,conformer_layers,max_pooling_factor, 
                      LEARNING_RATE,DEVICE)


    criterion = initialize_loss_function()

    # Initialize data loader
    train_loader = initialize_data_loader(train_data_path, train_labels_path,BATCH_SIZE, True, num_workers, prefetch_factor,pin_memory,apply_transform)
    train_labels_dict= load_json_dictionary(train_labels_path)

    LR_SCHEDULER = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # Set model to train
    PS_Model.train()

    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):

        dropout_prob = adjust_dropout_prob(PS_Model, epoch, NUM_EPOCHS)

        # Training step for the current epoch
        epoch_loss, utterance_predictions, files_names = train_one_epoch(
            PS_Model, train_loader, feature_extractor, optimizer, criterion,max_grad_norm,dropout_prob, DEVICE)

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"model_epochs{epoch + 1}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pth"
            save_checkpoint(PS_Model, optimizer, epoch + 1, os.path.join(model_save_path, model_filename))

        # Compute and log metrics
        utterance_labels = torch.tensor([train_labels_dict[file_name] for file_name in files_names])
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_metrics(utterance_predictions, utterance_labels)

        # Validation step (optional)
        if (epoch + 1) >= monitor_dev_epoch:
            dev_data_loader=initialize_data_loader(dev_data_path, dev_labels_path,BATCH_SIZE,False,num_workers, prefetch_factor,pin_memory)
            dev_labels_dict= load_json_dictionary(dev_labels_path)
            print(f"train_loader: {len(train_loader)} , dev_data_loader: {len(dev_data_loader)}")
            dev_metrics_dict = dev_one_epoch(PS_Model, feature_extractor,criterion,dev_data_loader, dev_labels_dict,dropout_prob,DEVICE)
            
            log_metrics_to_wandb(epoch, epoch_loss, utterance_eer, utterance_eer_threshold, dev_metrics_dict)               # Log metrics to W&B

            # Early stopping check
            # early_stopping(dev_metrics_dict['epoch_loss'], PS_Model)
            # if early_stopping.early_stop:
            #     print(f"Early stopping at epoch {epoch+1}")
            #     break

        else:
            log_metrics_to_wandb(epoch, epoch_loss, utterance_eer, utterance_eer_threshold, dev_metrics_dict= None)         # Log metrics to W&B

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

    # # Save segment_predictions, segment_labels, utterance_predictions, utterance_labels
    # torch.save(utterance_predictions,os.path.join(os.getcwd(),f'outputs/utterance_predictions_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pt'))
    # torch.save(torch.tensor(utterance_labels),os.path.join(os.getcwd(),f'outputs/utterance_labels_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pt'))

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

    # Define your paths and other fixed arguments
    BASE_DIR = os.getcwd()

    # Define training files and labels
    train_data_path=os.path.join(BASE_DIR,'database/train/con_wav')
    # train_data_path=os.path.join(BASE_DIR,'database/mini_database/train')
    train_labels_path=os.path.join(BASE_DIR,'database/utterance_labels/PartialSpoof_LA_cm_train_trl.json')
    dev_data_path=os.path.join(BASE_DIR, 'database/dev/con_wav')
    # dev_data_path=os.path.join(BASE_DIR, 'database/mini_database/dev')
    dev_labels_path=os.path.join(BASE_DIR, 'database/utterance_labels/PartialSpoof_LA_cm_dev_trl.json') 
    ssl_ckpt_path=os.path.join(os.getcwd(), 'models/w2v_large_lv_fsh_swbd_cv.pt')
    
    # Call train_model with parameters from W&B sweep
    train_model(train_data_path=train_data_path, 
               train_labels_path=train_labels_path,
               dev_data_path=dev_data_path, 
               dev_labels_path=dev_labels_path, 
               ssl_ckpt_path=ssl_ckpt_path,
               apply_transform=False,
               save_feature_extractor=True,
               feature_dim=768, 
               num_heads=8, 
               hidden_dim=128, 
               max_dropout=0.2,
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
               save_interval=10,
               model_save_path=os.path.join(os.getcwd(),'models/back_end_models'),
               patience=10,
               max_grad_norm=1.0,
               gamma=0.9,
               DEVICE=DEVICE)
    





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
    project_name='partial_spoof_Wav2Vec2_Conformer_binary_classifier'

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
            'NUM_EPOCHS': {'values': [60]},
            'LEARNING_RATE': {'values': [0.005]},
            # 'LEARNING_RATE': {'values': [0.00021195579137608126]},
            # 'LEARNING_RATE': {'values': [2.3550643486231242e-05]},
            'BATCH_SIZE': {'values': [8]},
            # 'CLASS0_WEIGHT': {'values': [0.42,0.45,0.48]},

        }
    }

    sweep_id = wandb.sweep(sweep=sweep_config,project=project_name)
    # sweep_id = wandb.sweep(sweep=sweep_config)
    wandb.agent(sweep_id, function=train, count=1)




if __name__ == "__main__":
    # Record the start time
    start_time = datetime.now()

    main()
    # inference()

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
    print(f"Total training time: {hours} hours, {minutes} minutes, {seconds} seconds")


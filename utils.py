from __future__ import absolute_import
from __future__ import print_function

import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as torch_nn
import torchaudio
import torch.nn.functional as torch_nn_func



def protocol_parse(protocol_filepath):
    """ Parse protocol of ASVspoof2019 and get bonafide/spoof for each trial
    
    input:
    -----
      protocol_filepath: string, path to the protocol file
        for convenience, I put train/dev/eval trials into a single protocol file
    
    output:
    -------
      data_buffer: dic, data_bufer[filename] -> 1 (bonafide), 0 (spoof)
    """ 
    data_buffer = {}
    temp_buffer = np.loadtxt(protocol_filepath, dtype='str')
    for row in temp_buffer:
        if row[-1] == 'bonafide':
            data_buffer[row[1]] = 1
        else:
            data_buffer[row[1]] = 0
    return data_buffer

def protocol_parse_con(reco2seglabel_filepath):
    """ Get label fro PartialSpoof database
    
    input:
    -----
      reco2seglabel_filepath: npy, path to the label file
    
    output:
    -------
      data_buffer: dict{list}, data_bufer[filename] -> 1 (bonafide), 0 (spoof)
    """ 
    data = np.load(reco2seglabel_filepath, allow_pickle=True)
    data_buffer=data.item()
    return data_buffer



# reco2seglabel_filepath="E:\projects\SCL-Deepfake-audio-detection-main\database\segment_labels\dev_seglab_0.01.npy"
# print(protocol_parse_con(reco2seglabel_filepath))


import torch
import torchaudio
import librosa  # For MFCC extraction

def preprocess_audio(audio_file, sample_rate=16000, frame_length=0.025, frame_shift=0.01, n_fft=512, n_mels=40, n_mfcc=20):
  """
  Preprocesses an audio file.

  Args:
    audio_file: Path to the audio file.
    sample_rate: Desired sample rate.
    frame_length: Length of each frame in seconds.
    frame_shift: Step size between frames in seconds.
    n_fft: Number of FFT points.
    n_mels: Number of Mel-frequency filter banks.
    n_mfcc: Number of MFCC coefficients.

  Returns:
    Preprocessed audio features.
  """

  # Load audio file
  waveform, sample_rate = torchaudio.load(audio_file)

  # Resample if necessary
  if waveform.shape[0] != 1:
    waveform = waveform.mean(dim=0, keepdim=True)
  if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

  # Convert to numpy array for librosa
  waveform = waveform.squeeze(0).numpy()

  # Extract MFCC features
  mfccs = librosa.feature.mfcc(y=waveform, sr=16000, n_fft=n_fft, n_mels=n_mels, n_mfcc=n_mfcc)

  # Normalize MFCCs (optional)
  # mfccs = librosa.util.normalize(mfccs)

  # Convert back to PyTorch tensor
  mfccs = torch.from_numpy(mfccs).float()

  # Additional preprocessing (e.g., delta and delta-delta features, feature normalization)

  return mfccs





import torch
import torchaudio

def preprocess_audio_lfcc(audio_file, sample_rate=16000, n_fft=512, n_lfcc=20):
  """
  Preprocesses an audio file using LFCC features.

  Args:
    audio_file: Path to the audio file.
    sample_rate: Desired sample rate.
    n_fft: Number of FFT points.
    n_lfcc: Number of LFCC coefficients.

  Returns:
    Preprocessed LFCC features.
  """

  # Load audio file
  waveform, sample_rate = torchaudio.load(audio_file)

  # Resample if necessary
  if waveform.shape[0] != 1:
    waveform = waveform.mean(dim=0, keepdim=True)
  if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

  # Convert to mono if necessary
  if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

  # Extract LFCC features
  # lfcc = torchaudio.transforms.LFCC(sample_rate=sample_rate, n_fft=n_fft, n_lfcc=n_lfcc)(waveform)
  lfcc = torchaudio.transforms.LFCC(sample_rate=sample_rate, n_lfcc=n_lfcc,speckwargs={"n_fft": n_fft})(waveform)

  return lfcc




# BASE_DIR=os.path.dirname(os.path.abspath(__file__))
# audio_file_path=os.path.join(BASE_DIR,'\database\eval\con_wav\CON_E_0000000.wav')

# Example usage
# mfccs_features = preprocess_audio(audio_file_path)
# print("mfccs_features.shape ",mfccs_features.shape)

# Example usage
# lfcc_features = preprocess_audio_lfcc(audio_file_path)
# print("lfcc_features.shape",lfcc_features.shape)



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
        # waveform = waveform / waveform.abs().max()

        # Method 2: Z-score normalization (mean=0, std=1)
        waveform = (waveform - waveform.mean()) / waveform.std()

        return waveform
    
    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.directory, file_name)

        try:
            waveform, sample_rate = torchaudio.load(file_path, normalize=True)
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None
        
        # Normalize waveform if needed
        # if self.normalize:
        #     waveform = self.normalize_waveform(waveform)

        # Apply any other transformations if provided
        if self.transform:
            waveform = self.transform(waveform)

        # Return raw waveform and sample rate
        file_name = file_name.split('.')[0]
        label = self.labels_dict.get(file_name).astype(int)

        label = torch.tensor(label, dtype=torch.int8)
        return {'waveform': waveform, 'sample_rate': sample_rate, 'label': label, 'file_name': file_name}
    




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
            waveform = pitch_shift(waveform)
        return waveform







def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Remove None values
    if len(batch) == 0:
        return None
    
    waveforms = [item['waveform'] for item in batch]
    labels = [item['label'] for item in batch]

    # Pad waveforms to have the same length
    waveforms_padded=pad_sequence([waveform.squeeze(0) for waveform in waveforms], batch_first=True)

    # Determine the maximum length of labels in the dataset
    max_label_length = 33

    # Pad labels to the fixed length of 33
    labels_padded = []
    for label in labels:
        # If the label is shorter than the fixed length, pad it
        if label.size(0) < max_label_length:
            padded_label = F.pad(label, (0, max_label_length - label.size(0)), value=-1)
            # padded_label = F.pad(label, (0, max_label_length - label.size(0)), value=float('nan'))
        else:
            padded_label = label[:max_label_length]
        labels_padded.append(padded_label)
    
    # Stack padded labels to a single tensor
    labels_padded = torch.stack(labels_padded)

    return {
        'waveform': waveforms_padded,
        'label': labels_padded,
        'sample_rate': [item['sample_rate'] for item in batch],
        'file_name': [item['file_name'] for item in batch]
    }





def compute_eer(predictions, labels):

    # Mask padding value
    predictions, labels =get_masked_labels_and_outputs(predictions, labels)
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





# from torch.utils.data import DataLoader
import torch.multiprocessing as mp

# Assuming AudioDataset and collate_fn are defined elsewhere
def get_raw_labeled_audio_data_loaders(directory, labels_dict, batch_size=32, shuffle=True, num_workers=0, prefetch_factor=None):
    
    # If multiprocessing is used, set start method to 'spawn' (for avoiding pickling issues)
    if num_workers > 0:
        mp.set_start_method('spawn', force=True)
    
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

# from torch.utils.data.distributed import DistributedSampler

# def get_raw_labeled_audio_data_loaders(rank,world_size,directory, labels_dict, batch_size=32, shuffle=True, num_workers=0, prefetch_factor=None):
    
#     # If multiprocessing is used, set start method to 'spawn' (for avoiding pickling issues)
#     if num_workers > 0:
#         mp.set_start_method('spawn', force=True)
    
#     # Create the dataset instance
#     dataset = RawLabeledAudioDataset(directory, labels_dict)
#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

#     # Create the DataLoader
#     data_loader = DataLoader(
#         dataset,
#         batch_size=batch_size, 
#         shuffle=shuffle, 
#         num_workers=num_workers, 
#         pin_memory=True,  # Enable page-locked memory for faster data transfer to GPU
#         prefetch_factor=prefetch_factor,  # How many batches to prefetch per worker
#         collate_fn=custom_collate_fn,  # Custom collate function to handle variable-length inputs
#         sampler=sampler
#     )
    
#     return data_loader






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


def create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_segment_eer,epoch_segment_eer_threshold,epoch_loss):
    metrics_dict=dict()
    metrics_dict['utterance_eer']=utterance_eer
    metrics_dict['utterance_eer_threshold']=utterance_eer_threshold
    metrics_dict['segment_eer']=epoch_segment_eer
    metrics_dict['segment_eer_threshold']=epoch_segment_eer_threshold
    metrics_dict['epoch_loss']=epoch_loss

    return metrics_dict



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





def plot_eer_per_epoch(NUM_EPOCHS, eer_per_epoch,save_dir=None):
    """
    Plot the Equal Error Rate (EER) per epoch.

    :param num_epochs: Number of epochs.
    :param eer_per_epoch: List or array of EER values for each epoch.
    """
    if len(eer_per_epoch) != NUM_EPOCHS:
        raise ValueError("Length of eer_per_epoch must be equal to num_epochs.")

    epochs = list(range(1, NUM_EPOCHS + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, eer_per_epoch, marker='o', linestyle='-', color='b')
    plt.title('Equal Error Rate (EER) per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('EER')
    plt.grid(True)
    plt.xticks(epochs)  # Show all epoch numbers
    plt.tight_layout()  # Adjust the plot to fit into the figure area.
    
    if save_dir:
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
    
        # Generate a unique filename based on hyperparameters
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        figure_filename = f"model_epochs{NUM_EPOCHS}_variable1_{timestamp}.png"

        # Construct the full path to save the file
        file_path = os.path.join(save_dir, figure_filename)
        # Save the figure
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")
    else:
        # Display the plot
        plt.show()

    # Close the plot to free up memory
    plt.close()



def plot_train_dev_eer_per_epoch(NUM_EPOCHS, train_eer_per_epoch, val_eer_per_epoch,save_dir=None):
    """
    Plot the Equal Error Rate (EER) per epoch for both training and validation.

    :param num_epochs: Number of epochs.
    :param train_eer_per_epoch: List or array of training EER values for each epoch.
    :param val_eer_per_epoch: List or array of validation EER values for each epoch.
    """
    if len(train_eer_per_epoch) != NUM_EPOCHS or len(val_eer_per_epoch) != NUM_EPOCHS:
        raise ValueError("Length of eer_per_epoch lists must be equal to num_epochs.")

    epochs = list(range(1, NUM_EPOCHS + 1))

    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_eer_per_epoch, marker='o', linestyle='-', color='b', label='Training EER')
    plt.plot(epochs, val_eer_per_epoch, marker='x', linestyle='--', color='r', label='Validation EER')
    plt.title('Equal Error Rate (EER) per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('EER')
    plt.grid(True)
    plt.xticks(epochs)  # Show all epoch numbers
    plt.legend()
    plt.tight_layout()  # Adjust the plot to fit into the figure area.
    
    if save_dir:
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
    
        # Generate a unique filename based on hyperparameters
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        figure_filename = f"model_epochs{NUM_EPOCHS}_variable2_{timestamp}.png"

        # Construct the full path to save the file
        file_path = os.path.join(save_dir, figure_filename)
        # Save the figure
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")
    else:
        # Display the plot
        plt.show()

    # Close the plot to free up memory
    plt.close()








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




def get_uttEER_by_seg(outputs,labels):
    """
    Measure Utterance EER based on segment-level score vectors.
    For each utteracne, using min() to choose the minimum segment score as the utterance score. 
    """

    mask_tensor = (labels != -1)
    # Initialize masked_output with a default value (e.g., NaN) to keep the same shape
    masked_outputs = outputs.clone()  # Copy the original output tensor
    # masked_outputs[~mask_tensor] = float('nan')  # Set invalid positions to NaN
    # masked_outputs[~mask_tensor] = float('inf')  # Set invalid positions to inf
    masked_outputs[~mask_tensor] = 64  # Set invalid positions to a placeholder value (e.g., 128), max number in int8
    # print(f"masked_output:\n {masked_outputs}")

    # If masked_outputs is 1D, just get the minimum value
    if masked_outputs.dim() == 1:
        # print("masked_outputs.dim() = 1")
        return torch.min(masked_outputs).unsqueeze(0)  # Return as 1D tensor

    # If masked_outputs is 2D, get the minimum across the specified dimension
    # return torch.max(masked_outputs, dim=1, keepdim=True).values
    return torch.min(masked_outputs, dim=1, keepdim=True).values



def count_files_in_directory(directory):
    # Get a list of all entries in the directory
    entries = os.listdir(directory)
    
    # Filter out only files (ignore directories)
    file_count = sum(1 for entry in entries if os.path.isfile(os.path.join(directory, entry)))
    
    return file_count




import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, model_output, labels_tensor,calculate_utt_loss=False,segment_weight=0.5, utt_weight=0.5):
        # Use BCEWithLogitsLoss with reduction='none' to compute loss per element
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        
        if calculate_utt_loss:
            masked_output, masked_labels=get_masked_labels_and_outputs(model_output,labels_tensor,keep_dims=True)
            utt_loss = loss_fn(masked_output.min(dim=1).values,masked_labels.min(dim=1).values)

            model_output,labels_tensor=get_masked_labels_and_outputs(model_output,labels_tensor)
            # Convert labels to float for BCEWithLogitsLoss
            labels_tensor = labels_tensor.float()
            segment_loss = loss_fn(model_output, labels_tensor)
            # return 0.5*segment_loss.mean()+0.5*utt_loss.mean()
            return segment_weight * segment_loss.mean() + utt_weight * utt_loss.mean()
        else:
            model_output,labels_tensor=get_masked_labels_and_outputs(model_output,labels_tensor)
            # Convert labels to float for BCEWithLogitsLoss
            labels_tensor = labels_tensor.float()
            segment_loss = loss_fn(model_output, labels_tensor)
            return segment_loss.mean()




def get_masked_labels_and_outputs(model_output, labels_tensor,keep_dims=False, mask_value=float('inf')):
    if keep_dims:
        # Create mask to identify valid labels (not -1)
        mask_tensor = (labels_tensor != -1)
        # Replace the -1 values in labels_tensor with a mask_value (e.g., NaN)
        masked_labels = labels_tensor.clone()  # Avoid modifying the original tensor
        masked_labels[~mask_tensor] = mask_value
        # Similarly, apply the mask to the model_output
        masked_output = model_output.clone()  # Avoid modifying the original tensor
        masked_output[~mask_tensor] = mask_value  # Replace invalid entries with mask_value
        return masked_output, masked_labels

    else:
        # Create mask to identify valid labels (not -1)
        mask_tensor = (labels_tensor != -1)
        # print(f"mask_tensor: {mask_tensor}")
        # Remove -1 values from labels using the mask
        masked_labels = labels_tensor[mask_tensor]
        # Remove equivalent positions in the output tensor using the mask
        masked_output = model_output[mask_tensor]
        return masked_output, masked_labels









class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False, path=os.path.join(os.getcwd(),'models/back_end_models/best_model.pth')):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float): Minimum change to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss - val_loss > self.delta and val_loss < 0.3:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
            self.best_model_wts = model.state_dict()  # Save best model weights
            torch.save(self.best_model_wts, self.path)  # Save the model checkpoint
        else:
            self.counter += 1
            if self.verbose:
                print(f'Validation loss did not improve. Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered.")

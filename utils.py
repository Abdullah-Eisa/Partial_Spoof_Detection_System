
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





class AudioDataset(Dataset):
    def __init__(self, directory, labels_dict, tokenizer,feature_extractor, transform=None, normalize=True):
        self.directory = directory
        self.labels_dict = labels_dict
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.normalize = normalize
        self.file_list = [f for f in os.listdir(directory) if f.endswith('.wav')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.directory, file_name)

        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None  # Or handle the error differently

        if self.transform:
            waveform = self.transform(waveform)

        # inputs = self.tokenizer(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = self.tokenizer(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt", padding="longest")
        # print(f'{file_name} has tokenized inputs of type {type(inputs)}  with size {inputs.size()}')
        # print(f'{file_name} has tokenized inputs of type {type(inputs)}')

        with torch.no_grad():
            for key in inputs:
                inputs[key] = inputs[key].to('cuda')

            # inputs = inputs().to('cuda')
            features = self.feature_extractor(**inputs).last_hidden_state.squeeze(0)
            # print(f'{file_name} has extracted features of type {type(features)}  with size {features.size()}')


        if self.normalize:
            features = F.normalize(features, dim=1)


        file_name=file_name.split('.')[0]

        # label = self.labels_dict.get(file_name, -1).astype(int)
        label = self.labels_dict.get(file_name).astype(int)
        # print(f'{file_name} has label of type {type(label)}  with size , Array={label}')
        label = torch.tensor(label, dtype=torch.int8).to('cuda')
        # print(f'{file_name} has label of type {type(label)}  with size {label.size()}, Array={label}')


        # Get file name from file path
        # filename_with_extension = file_path.split('/')[-1]  # Get the last part of the path
        # filename_without_extension = filename_with_extension.split('.')[0]  # Remove the extension
        # print(f'inside AudioDataset , file_name: {file_name} , filename_without_extension: {filename_without_extension} ')

        # return {'features': features, 'label': label, 'file_path': file_path}
        return {'features': features, 'label': label, 'file_name': file_name}
        # return {'features': features, 'label': label, 'file_path': filename_without_extension}






# # upsampled_labels
# import torch
# import torch.nn.functional as F

# def collate_fn(batch):
#     batch = [item for item in batch if item is not None]  # Remove None values
#     if len(batch) == 0:
#         return None

#     features = [item['features'] for item in batch]
#     labels = [item['label'] for item in batch]

#     # Pad features to have the same length
#     features_padded = pad_sequence(features, batch_first=True)

#     # Determine the maximum length of labels in the batch
#     # max_label_length = max(label.size(0) for label in labels)
#     max_label_length = 33

#     # Upsample labels to the maximum length using interpolation
#     labels_upsampled = []
#     for label in labels:
#         # Convert label to float for interpolation
#         label_float = label.float()  # Convert to float tensor
#         if label_float.size(0) < max_label_length:
#             # Calculate the scale factor
#             scale_factor = max_label_length / label_float.size(0)
#             # Upsample using interpolation
#             upsampled_label = F.interpolate(label_float.unsqueeze(0).unsqueeze(0), size=max_label_length, mode='linear', align_corners=True).squeeze(0).squeeze(0)
#         else:
#             upsampled_label = label_float
#         labels_upsampled.append(upsampled_label)

#     # Stack upsampled labels to a single tensor
#     labels_upsampled = torch.stack(labels_upsampled)

#     return {
#         'features': features_padded.to('cuda'),
#         'label': labels_upsampled.to('cuda'),
#         'file_name': [item['file_name'] for item in batch]
#     }



def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Remove None values
    if len(batch) == 0:
        return None
    
    features = [item['features'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Pad features to have the same length
    features_padded = pad_sequence(features, batch_first=True)

    # Determine the maximum length of labels in the batch
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
        'features': features_padded.to('cuda'),
        'label': labels_padded.to('cuda'),
        'file_name': [item['file_name'] for item in batch]
    }




def compute_det_curve(nontarget_scores,target_scores):
    # Flatten the input arrays to ensure they are 1D
    target_scores = np.ravel(target_scores)
    nontarget_scores = np.ravel(nontarget_scores)

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds

def compute_eer(nontarget_scores,target_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    
    # Mask padding value
    nontarget_scores,target_scores =get_masked_labels_and_outputs(nontarget_scores,target_scores)
    print(f"after Mask padding value,\n nontarget_scores=\n{nontarget_scores} target_scores=\n{target_scores} ")

    # Ensure scores and labels are PyTorch tensors and detach them
    nontarget_scores = nontarget_scores.detach().cpu().numpy()
    target_scores = target_scores.detach().cpu().numpy()

    # Check if inputs are multi-dimensional
    if target_scores.ndim > 1 and target_scores.shape[0] == nontarget_scores.shape[0]:
        num_labels = target_scores.shape[0]
        eer_results = []

        for i in range(num_labels):
            # Flatten scores for the i-th label
            score_i = target_scores[i, :]
            nontarget_i = nontarget_scores[i, :] if nontarget_scores.ndim > 1 else nontarget_scores

            # Compute EER for the i-th label
            try:
                frr, far, thresholds = compute_det_curve(nontarget_i,score_i)
                abs_diffs = np.abs(frr - far)
                min_index = np.argmin(abs_diffs)
                eer = np.mean((frr[min_index], far[min_index]))
                eer_results.append((eer, thresholds[min_index]))
            except Exception as e:
                print(f"Error computing EER for label {i}: {e}")
                continue

        if not eer_results:
            raise ValueError("No valid EER results found.")

        # Averaging EERs across all labels
        avg_eer = np.mean([eer for eer, _ in eer_results])
        avg_eer_threshold = np.mean([eer_threshold for _, eer_threshold in eer_results])
        
        return avg_eer, avg_eer_threshold

    else:
        # Single label case
        frr, far, thresholds = compute_det_curve(nontarget_scores,target_scores)
        abs_diffs = np.abs(frr - far)

        # Check for NaN values
        if np.any(np.isnan(frr)) or np.any(np.isnan(far)):
            raise ValueError("NaN values found in frr or far. Cannot compute EER.")

        min_index = np.argmin(abs_diffs)
        eer = np.mean((frr[min_index], far[min_index]))
        return eer, thresholds[min_index]













def get_audio_data_loaders(directory, labels_dict, tokenizer,feature_extractor, batch_size=32, shuffle=True, num_workers=0, prefetch_factor=None):
    dataset = AudioDataset(directory, labels_dict, tokenizer,feature_extractor)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        # pin_memory=True,  # Enable this for faster transfers
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn
        )
    
    return data_loader







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

def load_checkpoint(model, optimizer, path='checkpoint.pth'):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']




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
    masked_outputs[~mask_tensor] = 128  # Set invalid positions to a placeholder value (e.g., 128), max number in int8
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

    def forward(self, model_output, labels_tensor):
        # Convert labels to float for BCEWithLogitsLoss
        labels_tensor = labels_tensor.float()

        # Create mask: 1 for valid entries (not -1), 0 for padding (-1)
        mask_tensor = (labels_tensor != -1).float()

        # Use BCEWithLogitsLoss with reduction='none' to compute loss per element
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        loss_per_element = loss_fn(model_output, labels_tensor)

        # Apply mask to ignore padding positions
        masked_loss = loss_per_element * mask_tensor  # Zero out losses for padded positions

        # Calculate the mean of the masked loss for valid positions
        total_loss = masked_loss.sum()
        valid_elements_count = mask_tensor.sum()

        if valid_elements_count > 0:
            return total_loss / valid_elements_count
        else:
            return torch.tensor(0.0, device=model_output.device)  # Return 0 if no valid entries




def get_masked_labels_and_outputs(model_output,labels_tensor):
    # Create mask to identify valid labels (not -1)
    mask_tensor = (labels_tensor != -1)
    # print(f"mask_tensor:\n {mask_tensor}")
    # Remove -1 values from labels using the mask
    masked_labels = labels_tensor[mask_tensor]

    # Remove equivalent positions in the output tensor using the mask
    masked_output = model_output[mask_tensor]

    return masked_output,masked_labels 



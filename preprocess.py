import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from utils import *
import random
import torchaudio.transforms as T
from torch.utils.data import Dataset
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader , ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp

import os
import torch
import torchaudio
from torch.utils.data import Dataset


# ============================================================================================
# ============================================================================================
# ============================================================================================


class AudiosetDataset(Dataset):
    def __init__(self, directory, labels_dict, audio_conf=None, transform=True, normalize=True):
        """
        Args:
            directory (str): Path to the directory containing the audio files.
            labels_dict (dict): Dictionary of labels for each audio file.
            audio_conf (dict, optional): Audio configuration dictionary (e.g., num_mel_bins).
            transform (bool, optional): Whether to apply transformations like frequency/time masking.
            normalize (bool, optional): Whether to normalize the waveform. Default is True.
        """
        self.directory = directory
        self.labels_dict = labels_dict
        self.audio_conf = audio_conf
        self.transform = transform
        self.normalize = normalize
        self.file_list = [f for f in os.listdir(directory) if f.endswith('.wav')]

    def __len__(self):
        return len(self.file_list)

    def normalize_waveform(self, waveform):
        """
        Normalize the waveform by scaling it to [-1, 1].
        """
        # print(f"waveform.mean(),waveform.std() = {waveform.mean(),waveform.std()}")
        return waveform / waveform.abs().max()
        # Method 2: Z-score normalization (mean=0, std=1)
        # return (waveform - waveform.mean()) / waveform.std()

    def _wav2fbank(self, file_path):
        """
        Convert a WAV file to its Mel-frequency bank features.
        """
        waveform, sr = torchaudio.load(file_path, normalize=False)
        if self.normalize:
            waveform = self.normalize_waveform(waveform)

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr,
            use_energy=False, window_type='hanning', num_mel_bins=self.audio_conf.get('num_mel_bins'),
            dither=0.0, frame_shift=10
        )

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]
        p = target_length - n_frames

        # Cut and pad if necessary
        # if p > 0:
        #     fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
        # elif p < 0:
        #     fbank = fbank[:target_length, :]

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]


        return fbank

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.
        """
        # if isinstance(idx, tuple):  # Ensure idx is an integer, not a tuple
        #     raise TypeError(f"Expected integer index, got tuple: {idx}")

        file_name = self.file_list[idx]
        file_path = os.path.join(self.directory, file_name)

        try:
            fbank = self._wav2fbank(file_path)
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None

        # Apply transformations (if specified)
        if self.transform:
            freqm = torchaudio.transforms.FrequencyMasking(self.audio_conf.get('freqm', 0))
            timem = torchaudio.transforms.TimeMasking(self.audio_conf.get('timem', 0))
            fbank = torch.transpose(fbank, 0, 1)  # Adjust for the new torchaudio version
            if self.audio_conf.get('freqm') > 0:
                fbank = freqm(fbank)
            if self.audio_conf.get('timem') > 0:
                fbank = timem(fbank)
            fbank = torch.transpose(fbank, 0, 1)  # Revert transpose

        # Normalize the fbank if required
        if self.normalize:
            fbank = self.normalize_waveform(fbank)

        # Return raw waveform and sample rate
        file_name = file_name.split('.')[0]
        # label = self.labels_dict.get(file_name).astype(int)
        label = self.labels_dict.get(file_name)
        # label = torch.tensor(label, dtype=torch.int8)
        label = torch.tensor(label)

        # print(type(fbank),type(label),type(file_name))
        return {'fbank': fbank, 'label': label, 'file_name': file_name}




def initialize_data_loader(data_path, labels_path, audio_conf, BATCH_SIZE=32, shuffle=False, num_workers=0, prefetch_factor=None,pin_memory=False):
    """Initialize and return the training data loader"""
    labels_dict= load_json_dictionary(labels_path)

        # If multiprocessing is used, set start method to 'spawn' (for avoiding pickling issues)
    if num_workers > 0:
        if os.name == 'nt':  # Windows
            mp.set_start_method('spawn', force=True)
        else:  # Unix-based (Linux, macOS, etc.)
            mp.set_start_method('fork', force=True)
    
    # Create the dataset instance
    combined_dataset = AudiosetDataset(data_path, labels_dict,audio_conf)
    
    # Create the DataLoader
    return DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory,  # Enable page-locked memory for faster data transfer to GPU
        prefetch_factor=prefetch_factor  # How many batches to prefetch per worker
    )
    




if __name__ == "__main__":

    # Define the audio configuration
    audio_conf = {
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

    # Create an instance of the dataset
    # label_csv = "path_to_label_csv.csv"
    # dataset_json_file = "path_to_dataset_json_file.json"
    # Define training files and labels

    # train_data_path=os.path.join(os.getcwd(),'database/train/con_wav')
    # train_labels_path=os.path.join(os.getcwd(),'database/utterance_labels/PartialSpoof_LA_cm_train_trl.json')

    # Sample with the largest fbank size:
    # File name: CON_T_0010513
    # fbank size: torch.Size([2100, 128])
    # Label: 0
    # Total elements: torch.Size([2100, 128])

    # train_data_path=os.path.join(os.getcwd(),'database/dev/con_wav')
    # train_labels_path=os.path.join(os.getcwd(),'database/utterance_labels/PartialSpoof_LA_cm_dev_trl.json')

    # Sample with the largest fbank size:
    # File name: CON_D_0016975
    # fbank size: torch.Size([1532, 128])
    # Label: 0
    # Total elements: torch.Size([1532, 128])

    train_data_path=os.path.join(os.getcwd(),'database/eval/con_wav')
    train_labels_path=os.path.join(os.getcwd(),'database/utterance_labels/PartialSpoof_LA_cm_eval_trl.json')
    # Sample with the largest fbank size:
    # File name: CON_E_0051069
    # fbank size: torch.Size([1818, 128])
    # Label: 0
    # Total elements: torch.Size([1818, 128])

    from utils import load_json_dictionary
    labels_dict= load_json_dictionary(train_labels_path)

    dataset = AudiosetDataset(train_data_path,labels_dict, audio_conf)

    sample = dataset[0]  # Correct usage, passing an integer index
    print(f"fbank size= {sample['fbank'].size()}, label= {sample['label']}")

    fbank_sizes = []

    max_size=sample['fbank'].size()
    for i in range(len(dataset)):
        sample = dataset[i]
        fbank_size = sample['fbank'].size()  # Get the size of the fbank tensor
        # fbank_size_num_elements = torch.numel(sample['fbank'])  # Get the total number of elements in the fbank tensor
        fbank_sizes.append(fbank_size)  # Store the size

        if fbank_size[0] > max_size[0]:
            max_size = fbank_size
            max_fbank_sample = sample

    # After the loop, print the sample with the largest fbank size
    if max_fbank_sample is not None:
        print(f"Sample with the largest fbank size:")
        print(f"File name: {max_fbank_sample['file_name']}")
        print(f"fbank size: {max_fbank_sample['fbank'].size()}")
        print(f"Label: {max_fbank_sample['label']}")
        print(f"Total elements: {max_size}")



    import matplotlib.pyplot as plt
    import seaborn as sns

    # Extract the time frames and frequency bins for all fbank tensors
    time_frames = [size[0] for size in fbank_sizes]
    frequency_bins = [size[1] for size in fbank_sizes]

    # Plot the distribution of time frames and frequency bins
    plt.figure(figsize=(12, 6))

    # Plot distribution of time frames
    plt.subplot(1, 2, 1)
    sns.histplot(time_frames, bins=30, kde=True, color='blue', edgecolor='black')
    plt.title('Distribution of Time Frames')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency')

    # Plot distribution of frequency bins
    plt.subplot(1, 2, 2)
    sns.histplot(frequency_bins, bins=30, kde=True, color='green', edgecolor='black')
    plt.title('Distribution of Frequency Bins')
    plt.xlabel('Frequency Bins')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Optional: Visualize as a scatter plot (time frames vs frequency bins)
    plt.figure(figsize=(8, 6))
    plt.scatter(time_frames, frequency_bins, alpha=0.5, color='purple')
    plt.title('Time Frames vs Frequency Bins')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.show()

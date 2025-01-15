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
import librosa

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
# ============================================================================================
# ============================================================================================


class ASVspoof2019(Dataset):
    def __init__(self, data_path,labels_path, transform=None,normalize=True, label_map = {"spoof": 1, "bonafide": 0}):
        super(ASVspoof2019, self).__init__()

        self.data_path = data_path
        self.labels_path = labels_path
        self.transform = transform
        self.normalize = normalize
        self.label_map = label_map

        self.all_files = librosa.util.find_files(self.data_path)
        self.all_labels= self._get_labels()


    def __len__(self):
        return len(self.all_files)

    def _get_label(self, file_name):
        # return self.label_map[file_name.split("_")[1]]
        return self.label_map[file_name]
    
    def _get_labels(self):
        labels_path= self.labels_path 
            
        # labels_dict = {}
        labels_dict = dict()
        # file_list=[]
        with open(labels_path, 'r') as f:
            file_lines = f.readlines()

        for line in file_lines:
            _, key,_,_,label = line.strip().split(' ')
            labels_dict[key] = self._get_label(label)
        
        return labels_dict


    def _normalize_waveform(self, waveform):
        """
        Normalize the waveform by scaling it to [-1, 1] or applying Z-score normalization.
        Args: waveform (Tensor): The input waveform tensor.
        Returns: Tensor: The normalized waveform.
        """
        # Method 1: Normalize to [-1, 1]
        waveform = waveform / waveform.abs().max()
        # Method 2: Z-score normalization (mean=0, std=1)
        # waveform = (waveform - waveform.mean()) / waveform.std()
        return waveform
    
    def __getitem__(self, idx):
        file_path = self.all_files[idx]
        base_name = os.path.basename(file_path)
        file_name = base_name.split(".")[0]

        try:
            # waveform, sample_rate = torchaudio.load(file_path, normalize=True)
            waveform, sample_rate = torchaudio.load(file_path, normalize=False)
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None
        
        # Normalize waveform if needed
        if self.normalize:
            waveform = self._normalize_waveform(waveform)

        # Apply any other transformations if provided
        if self.transform:
            waveform = self.transform(waveform)

        # label = self.labels_dict.get(file_name)
        label = self.all_labels.get(file_name,0)
        # label = torch.tensor(label, dtype=torch.int8)
        label = torch.tensor(label)

        return {'waveform': waveform, 'sample_rate': sample_rate, 'label': label, 'file_name': file_name}
    
    # def collate_fn(self, samples):
    #     return default_collate(samples)
    
    



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



def initialize_data_loader(data_path, labels_path,BATCH_SIZE=32, shuffle=True, num_workers=0, prefetch_factor=None,pin_memory=False,apply_transform=False):
    """Initialize and return the training data loader"""

        # If multiprocessing is used, set start method to 'spawn' (for avoiding pickling issues)
    if num_workers > 0:
        if os.name == 'nt':  # Windows
            mp.set_start_method('spawn', force=True)
        else:  # Unix-based (Linux, macOS, etc.)
            mp.set_start_method('fork', force=True)
    
    # Create the dataset instance
    combined_dataset = ASVspoof2019(data_path, labels_path)
    
    if apply_transform:
        # Apply pitch shift transform
        pitch_shift_transform = PitchShiftTransform(sample_rate=16000, pitch_shift_prob=1.0, pitch_shift_steps=(-2, 2))

        # # Initialize the dataset with the transform
        augmented_dataset = ASVspoof2019(
            directory=data_path,
            labels_dict=labels_path,
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
    


if __name__ == "__main__":

    train_data_path=os.path.join(os.getcwd(),'database/train/con_wav')
    # train_labels_path=os.path.join(os.getcwd(),'database/utterance_labels/PartialSpoof_LA_cm_train_trl.json')
    train_labels_path=os.path.join(os.getcwd(),'database/utterance_labels/ASVspoof2019.LA.cm.train.trl.txt')

    # train_data_path=os.path.join(os.getcwd(),'database/dev/con_wav')
    # train_labels_path=os.path.join(os.getcwd(),'database/utterance_labels/PartialSpoof_LA_cm_dev_trl.json')

    # train_data_path=os.path.join(os.getcwd(),'database/eval/con_wav')
    # train_labels_path=os.path.join(os.getcwd(),'database/utterance_labels/PartialSpoof_LA_cm_eval_trl.json')

    dataset = ASVspoof2019(train_data_path,train_labels_path)
    print(f"len(dataset): {len(dataset)}")
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"sample: {sample['waveform'].shape}, {sample['label']}")
        if i==50:
            break


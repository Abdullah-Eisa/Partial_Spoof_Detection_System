
import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # For progress bar (optional)



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
# ============================================================================================
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



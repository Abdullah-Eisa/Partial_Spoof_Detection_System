import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from utils.utils import *
import random
import torchaudio.transforms as T
from torch.utils.data import Dataset
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader , ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp
import librosa
from typing import Dict, List

# Data augmentation transforms for Audio files
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
# ASVspoof2019 Dataset class

class ASVspoof2019(Dataset):
    def __init__(self, data_path,labels_path, transform=None,normalize=True, label_map = {"spoof": 1, "bonafide": 0}):
        super(ASVspoof2019, self).__init__()

        self.data_path = data_path
        self.labels_path = labels_path
        self.transform = transform
        self.normalize = normalize
        self.label_map = label_map

        self.all_files = librosa.util.find_files(self.data_path)

        # num_files = 1000
        # self.all_files = random.sample(
        #     self.all_files,
        #     min(num_files, len(self.all_files))
        # )


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
        # print("file_lines= ",file_lines)
        for line in file_lines:
            # print("line= ",line.strip())
            line = line.strip()
            if not line:continue  # Skip empty lines

            try:
                _, key, _, _, label = line.split(' ')
                labels_dict[key] = self._get_label(label)
            except ValueError:
                # If there are not exactly 5 values, print a warning
                print(f"Warning: Skipping malformed line: {line}")
        
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
        label = standardize_labels("ASVspoof2019_LA_Dataset", label)
        label = torch.tensor(label)

        return {'waveform': waveform, 'sample_rate': sample_rate, 'label': label, 'file_name': file_name}
    
    # def collate_fn(self, samples):
    #     return default_collate(samples)
        
    

# PartialSpoof Dataset class
class PartialSpoofDataset(Dataset):
    def __init__(self, directory,labels_path, transform=None,normalize=True):
        """
        Args:
            directory (str): Path to the directory containing the audio files.
            labels_path (str): Path to labels for all audio files.
            save_dir (str): Path to the directory where the extracted features will be saved.
            transform (callable, optional): Optional transform to apply to the waveform.
            normalize (bool, optional): Whether to normalize the waveform. Default is True.
        """
        super(PartialSpoofDataset, self).__init__()


        self.directory = directory
        self.labels_path = labels_path
        self.labels_dict = self._get_labels()
        self.transform = transform
        self.normalize = normalize
        self.file_list = [f for f in os.listdir(directory) if f.endswith('.wav')]
        # self.file_list = [f for f in os.listdir(directory) if f.endswith('.wav')][:1000]

        # Ensure the save directory exists
        # os.makedirs(save_dir, exist_ok=True)

    def __len__(self):
        return len(self.file_list)

    def _get_labels(self):
        return load_json_dictionary(self.labels_path)

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
        label = standardize_labels("PartialSpoof_Dataset", label)
        label = torch.tensor(label)

        return {'waveform': waveform, 'sample_rate': sample_rate, 'label': label, 'file_name': file_name}
    






# RFP Dataset class
class RFP_Dataset(Dataset):
    def __init__(self, data_path,labels_path, transform=None,normalize=True, label_map = {"spoof": 1, "genuine": 0}):
        super(RFP_Dataset, self).__init__()

        self.data_path = data_path
        self.labels_path = labels_path
        self.transform = transform
        self.normalize = normalize
        self.label_map = label_map

        self.all_files = librosa.util.find_files(self.data_path)

        # num_files = 1000
        # self.all_files = random.sample(
        #     self.all_files,
        #     min(num_files, len(self.all_files))
        # )
        # self.all_labels= self._get_labels()


    def __len__(self):
        return len(self.all_files)

    def _get_label(self, file_name):
        # return self.label_map[file_name.split("_")[1]]
        return self.label_map[file_name]
    
    def _get_labels(self):
        labels_path= self.labels_path 
            
        labels_dict = dict()
        with open(labels_path, 'r') as f:
            file_lines = f.readlines()
        for line in file_lines:
            line = line.strip()
            if not line:continue  # Skip empty lines

            try:
                _, key, _, _, label = line.split(' ')
                labels_dict[key] = self._get_label(label)
            except ValueError:
                print(f"Warning: Skipping malformed line: {line}")
        
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

        # # label = self.all_labels.get(file_name,0)
        # label = self.all_labels.get(file_name)
        try:
            # label = self.all_labels.get(file_name, 0)
            label = self.all_labels.get(file_name)
        except Exception as e:
            print(f"Error while processing file: {file_name}")
            print(f"Exception: {e}")
            label = self.all_labels.get(file_name, 0)   # or whatever fallback value you prefer


        label = standardize_labels("RFP_Dataset", label)
        label = torch.tensor(label)

        return {'waveform': waveform, 'sample_rate': sample_rate, 'label': label, 'file_name': file_name}
    
    # def collate_fn(self, samples):
    #     return default_collate(samples)




# ============================================================================================
# ============================================================================================
# ============================================================================================



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



def initialize_data_loader(dataset_name,data_path, labels_path,BATCH_SIZE=32, shuffle=True, num_workers=0, prefetch_factor=None,pin_memory=False,apply_transform=False):
    """Initialize and return the training data loader"""

        # If multiprocessing is used, set start method to 'spawn' (for avoiding pickling issues)
    if num_workers > 0:
        if os.name == 'nt':  # Windows
            mp.set_start_method('spawn', force=True)
        else:  # Unix-based (Linux, macOS, etc.)
            mp.set_start_method('fork', force=True)
    
    # Choose the dataset to train on and create the DataLoader
    if dataset_name == "RFP_Dataset":
        print("You selected RFP_Dataset.")
        audio_dataset = RFP_Dataset(data_path, labels_path)

    elif dataset_name == "PartialSpoof_Dataset":
        print("You selected PartialSpoof_Dataset.")
        audio_dataset = PartialSpoofDataset(data_path, labels_path)

    elif dataset_name == "ASVspoof2019_LA_Dataset":
        print("You selected ASVspoof2019_LA_Dataset.")
        audio_dataset = ASVspoof2019(data_path, labels_path)

    else:
        print("Invalid dataset name selected.")



    # Create the DataLoader
    return DataLoader(
        audio_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory,  # Enable page-locked memory for faster data transfer to GPU
        prefetch_factor=prefetch_factor,  # How many batches to prefetch per worker
        collate_fn=custom_collate_fn  # Custom collate function to handle variable-length inputs
    )
    


def initialize_multi_dataset_loader(
    datasets_config: List[Dict],
    BATCH_SIZE=32,
    shuffle=True,
    num_workers=0,
    prefetch_factor=None,
    pin_memory=False,
    # WeightedRandomSampling=False
    WeightedRandomSampling=True
):
    """
    Initialize DataLoader for multiple datasets combined.
    
    Args:
        datasets_config: List of dicts with keys:
            - dataset_name: str
            - data_path: str
            - labels_path: str
        BATCH_SIZE: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        prefetch_factor: Prefetch factor
        pin_memory: Pin memory for GPU
    
    Returns:
        Combined DataLoader
    
    Example:
        >>> datasets_config = [
        ...     {
        ...         'dataset_name': 'RFP_Dataset',
        ...         'data_path': 'database/RFP/training',
        ...         'labels_path': 'database/RFP/labels/train.txt'
        ...     },
        ...     {
        ...         'dataset_name': 'PartialSpoof_Dataset',
        ...         'data_path': 'database/PartialSpoof/train/con_wav',
        ...         'labels_path': 'database/PartialSpoof/labels/train.json'
        ...     }
        ... ]
        >>> loader = initialize_multi_dataset_loader(datasets_config, BATCH_SIZE=8)
    """
    import torch.multiprocessing as mp
    from torch.utils.data import ConcatDataset
    
    # Set multiprocessing start method
    if num_workers > 0:
        if os.name == 'nt':  # Windows
            mp.set_start_method('spawn', force=True)
        else:  # Unix
            mp.set_start_method('fork', force=True)
    
    # Create individual datasets
    all_datasets = []
    
    for config in datasets_config:
        dataset_name = config['dataset_name']
        data_path = config['data_path']
        labels_path = config['labels_path']
        
        print(f"Loading {dataset_name}...")
        
        if dataset_name == "RFP_Dataset":
            dataset = RFP_Dataset(data_path, labels_path)
        
        elif dataset_name == "PartialSpoof_Dataset":
            dataset = PartialSpoofDataset(data_path, labels_path)
        
        elif dataset_name == "ASVspoof2019_LA_Dataset":
            dataset = ASVspoof2019(data_path, labels_path)
        
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        all_datasets.append(dataset)
        print(f"  Loaded {len(dataset)} samples")
    
    # Combine datasets
    combined_dataset = ConcatDataset(all_datasets)
    total_samples = len(combined_dataset)
    
    print(f"\nCombined Dataset Statistics:")
    print(f"  Total samples: {total_samples}")
    for i, dataset in enumerate(all_datasets):
        print(f"  {datasets_config[i]['dataset_name']}: {len(dataset)} "
              f"({len(dataset)/total_samples*100:.1f}%)")
    


    if WeightedRandomSampling:
        
        from torch.utils.data import WeightedRandomSampler
        # Calculate class weights
        dataset_sizes = [len(d) for d in all_datasets]
        weights = [1.0 / size for size in dataset_sizes]
        sample_weights = []

        for i, dataset in enumerate(all_datasets):
            sample_weights.extend([weights[i]] * len(dataset))

        print(f"Sample weights distribution: {sample_weights[:10]}...")

        sampler = WeightedRandomSampler(
            sample_weights, 
            num_samples=len(combined_dataset),
            replacement=True
        )

        # Modify return statement:
        return DataLoader(
            combined_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,  # Use sampler instead of shuffle
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            collate_fn=custom_collate_fn
        )

    else:
        # Create DataLoader
        return DataLoader(
            combined_dataset,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            collate_fn=custom_collate_fn
        )



def standardize_labels(dataset_name, label):
    """
    Standardize labels across datasets to consistent format.
    Convention: 0 = bonafide/genuine, 1 = spoof
    
    Args:
        dataset_name: Name of the dataset
        label: Original label from dataset
    
    Returns:
        Standardized label (0 or 1)
    """
    if dataset_name == "PartialSpoof_Dataset":
        # PartialSpoof: 0=spoof, 1=bonafide (INVERTED!)
        return 1 - label  # Flip: 0→1, 1→0
    
    elif dataset_name in ["ASVspoof2019_LA_Dataset", "RFP_Dataset"]:
        # ASVspoof & RFP: 0=bonafide, 1=spoof (CORRECT)
        return label
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

if __name__ == "__main__":
    # Test the ASVspoof2019 dataset
    train_data_path=os.path.join(os.getcwd(),'database/ASVspoof2019/LA/ASVspoof2019_LA_train/flac')
    train_labels_path=os.path.join(os.getcwd(),'database/utterance_labels/ASVspoof2019.LA.cm.train.trl.txt')

    dataset = ASVspoof2019(train_data_path,train_labels_path)
    print(f"len(dataset): {len(dataset)}")
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"sample: {sample['waveform'].shape}, {sample['label']}")
        if i==50:
            break


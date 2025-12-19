import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np


class FeatureExtractorFactory:
    """Factory class to create different feature extractors"""
    
    @staticmethod
    def create_extractor(config, device='cpu'):
        """
        Create feature extractor based on config
        
        Args:
            config: Configuration dictionary
            device: Device to load model on
            
        Returns:
            Feature extractor instance
        """
        extractor_type = config['feature_extractor']['type'].lower()
        
        if extractor_type == 'wav2vec2':
            return Wav2Vec2Extractor(
                checkpoint_path=config['feature_extractor']['ssl_checkpoint'],
                device=device
            )
        elif extractor_type == 'hubert':
            return HuBERTExtractor(
                checkpoint_path=config['feature_extractor']['ssl_checkpoint'],
                device=device
            )
        elif extractor_type == 'mfcc':
            return MFCCExtractor(
                n_mfcc=config['feature_extractor'].get('mfcc_n_mfcc', 40),
                log_mels=config['feature_extractor'].get('mfcc_log_mels', True),
                device=device
            )
        elif extractor_type == 'lfcc':
            return LFCCExtractor(
                n_filters=config['feature_extractor'].get('lfcc_n_filters', 20),
                n_lfcc=config['feature_extractor'].get('lfcc_n_lfcc', 40),
                device=device
            )
        else:
            raise ValueError(f"Unknown feature extractor type: {extractor_type}")


class Wav2Vec2Extractor(nn.Module):
    """Wav2Vec 2.0 feature extractor"""
    
    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__()
        self.device = device
        self.model = torch.hub.load('s3prl/s3prl', 'wav2vec2', 
                                    model_path=checkpoint_path).to(device)
        self.model.eval()
        
    # def forward(self, waveforms):
    #     """
    #     Extract features from waveforms
        
    #     Args:
    #         waveforms: (batch_size, time) tensor
            
    #     Returns:
    #         features: (batch_size, time_frames, feature_dim) tensor
    #     """
    #     with torch.no_grad():
    #         features = self.model(waveforms)['hidden_states'][-1]
    #     return features
    
    def forward(self, waveforms):
        """
        Extract features from waveforms
        
        Args:
            waveforms: (batch_size, time) tensor
            
        Returns:
            dict with 'hidden_states' key containing features
        """
        with torch.no_grad():
            # The s3prl model returns a dict, we keep it consistent
            output = self.model(waveforms)
            
            # Ensure output is in the expected format
            if not isinstance(output, dict):
                # If it returns tensor directly, wrap it
                output = {'hidden_states': [output]}
                
        return output    



    def get_feature_dim(self):
        # return 1024  # Wav2Vec 2.0 large has 1024 dims
        return 768  # Wav2Vec 2.0 large has 1024 dims
    
    def get_output_dim_after_pooling(self, max_pooling_factor):
        """Calculate output dimension after max pooling"""
        if max_pooling_factor is None:
            return self.get_feature_dim()
        return self.get_feature_dim() // max_pooling_factor


class HuBERTExtractor(nn.Module):
    """HuBERT feature extractor"""
    
    def __init__(self, checkpoint_path, device='cpu'):
        super().__init__()
        self.device = device
        # Load HuBERT model from s3prl
        self.model = torch.hub.load('s3prl/s3prl', 'hubert', 
                                    model_path=checkpoint_path).to(device)
        self.model.eval()
        
    # def forward(self, waveforms):
    #     """
    #     Extract features from waveforms
        
    #     Args:
    #         waveforms: (batch_size, time) tensor
            
    #     Returns:
    #         features: (batch_size, time_frames, feature_dim) tensor
    #     """
    #     with torch.no_grad():
    #         features = self.model(waveforms)['hidden_states'][-1]
    #     return features
    
    def forward(self, waveforms):
        """
        Extract features from waveforms
        
        Args:
            waveforms: (batch_size, time) tensor
            
        Returns:
            dict with 'hidden_states' key containing features
        """
        with torch.no_grad():
            # The s3prl model returns a dict, we keep it consistent
            output = self.model(waveforms)
            
            # Ensure output is in the expected format
            if not isinstance(output, dict):
                # If it returns tensor directly, wrap it
                output = {'hidden_states': [output]}
                
        return output    


    def get_feature_dim(self):
        return 1024  # HuBERT large has 1024 dims
    
    def get_output_dim_after_pooling(self, max_pooling_factor):
        """Calculate output dimension after max pooling"""
        if max_pooling_factor is None:
            return self.get_feature_dim()
        return self.get_feature_dim() // max_pooling_factor


class MFCCExtractor(nn.Module):
    """MFCC feature extractor"""
    
    def __init__(self, n_mfcc=40, sample_rate=16000, n_fft=400, 
                 hop_length=160, n_mels=80, log_mels=True, device='cpu'):
        super().__init__()
        self.device = device
        self.n_mfcc = n_mfcc
        self.log_mels = log_mels
        
        # Create MFCC transform
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels,
            }
        ).to(device)
        
    def forward(self, waveforms):
        """
        Extract MFCC features from waveforms
        
        Args:
            waveforms: (batch_size, time) tensor
            
        Returns:
            features: (batch_size, time_frames, n_mfcc) tensor
        """
        # Ensure waveforms are 2D: (batch_size, time)
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)
        
        # Compute MFCCs: (batch_size, n_mfcc, time_frames)
        mfccs = self.mfcc_transform(waveforms)
        
        # Transpose to (batch_size, time_frames, n_mfcc)
        features = mfccs.transpose(1, 2)
        
        return features
    
    def get_feature_dim(self):
        return self.n_mfcc
    
    def get_output_dim_after_pooling(self, max_pooling_factor):
        """Calculate output dimension after max pooling"""
        if max_pooling_factor is None:
            return self.get_feature_dim()
        return self.get_feature_dim() // max_pooling_factor


class LFCCExtractor(nn.Module):
    """LFCC (Linear Frequency Cepstral Coefficients) feature extractor"""
    
    def __init__(self, n_filters=20, n_lfcc=40, sample_rate=16000, 
                 n_fft=512, hop_length=160, device='cpu'):
        super().__init__()
        self.device = device
        self.n_filters = n_filters
        self.n_lfcc = n_lfcc
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Create spectrogram transform
        self.spectrogram = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0
        ).to(device)
        
        # Create linear filterbank
        self.linear_filterbank = self._create_linear_filterbank(
            n_filters, n_fft, sample_rate
        ).to(device)
        
        # DCT matrix for cepstral coefficients
        self.dct_matrix = self._create_dct_matrix(n_lfcc, n_filters).to(device)
        
    def _create_linear_filterbank(self, n_filters, n_fft, sample_rate):
        """Create linear-spaced filterbank"""
        # Frequency bins
        fft_freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        
        # Linear-spaced filter centers
        linear_freqs = torch.linspace(0, sample_rate / 2, n_filters + 2)
        
        # Create triangular filters
        filterbank = torch.zeros(n_filters, n_fft // 2 + 1)
        
        for i in range(n_filters):
            left = linear_freqs[i]
            center = linear_freqs[i + 1]
            right = linear_freqs[i + 2]
            
            # Rising slope
            rising = (fft_freqs >= left) & (fft_freqs <= center)
            filterbank[i, rising] = (fft_freqs[rising] - left) / (center - left)
            
            # Falling slope
            falling = (fft_freqs >= center) & (fft_freqs <= right)
            filterbank[i, falling] = (right - fft_freqs[falling]) / (right - center)
        
        return filterbank
    
    def _create_dct_matrix(self, n_lfcc, n_filters):
        """Create DCT transformation matrix"""
        dct_matrix = torch.zeros(n_lfcc, n_filters)
        for i in range(n_lfcc):
            for j in range(n_filters):
                dct_matrix[i, j] = np.cos(np.pi * i * (j + 0.5) / n_filters)
            if i == 0:
                dct_matrix[i, :] *= np.sqrt(1.0 / n_filters)
            else:
                dct_matrix[i, :] *= np.sqrt(2.0 / n_filters)
        return dct_matrix
    
    def forward(self, waveforms):
        """
        Extract LFCC features from waveforms
        
        Args:
            waveforms: (batch_size, time) tensor
            
        Returns:
            features: (batch_size, time_frames, n_lfcc) tensor
        """
        # Ensure waveforms are 2D: (batch_size, time)
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)
        
        batch_size = waveforms.shape[0]
        
        # Compute power spectrogram: (batch_size, freq_bins, time_frames)
        spec = self.spectrogram(waveforms)
        
        # Apply linear filterbank: (batch_size, n_filters, time_frames)
        filtered = torch.matmul(self.linear_filterbank, spec)
        
        # Log compression
        filtered = torch.log(filtered + 1e-8)
        
        # Apply DCT: (batch_size, n_lfcc, time_frames)
        lfcc = torch.matmul(self.dct_matrix, filtered)
        
        # Transpose to (batch_size, time_frames, n_lfcc)
        features = lfcc.transpose(1, 2)
        
        return features
    
    def get_feature_dim(self):
        return self.n_lfcc
    
    def get_output_dim_after_pooling(self, max_pooling_factor):
        """Calculate output dimension after max pooling"""
        if max_pooling_factor is None:
            return self.get_feature_dim()
        return self.get_feature_dim() // max_pooling_factor


def get_feature_dim_from_config(config):
    """
    Get the feature dimension based on the feature extractor type
    
    Args:
        config: Configuration dictionary
        
    Returns:
        feature_dim: Integer feature dimension
    """
    extractor_type = config['feature_extractor']['type'].lower()
    
    if extractor_type == 'wav2vec2':
        return 768  # SSL models typically output 1024 dims
    elif extractor_type == 'hubert':
        return 1024
    elif extractor_type == 'mfcc':
        return config['feature_extractor'].get('mfcc_n_mfcc', 40)
    elif extractor_type == 'lfcc':
        return config['feature_extractor'].get('lfcc_n_lfcc', 40)
    else:
        raise ValueError(f"Unknown feature extractor type: {extractor_type}")


def calculate_conformer_input_dim(base_feature_dim, max_pooling_factor, num_heads):
    """
    Calculate the conformer input dimension that is divisible by num_heads
    
    Args:
        base_feature_dim: Base feature dimension from extractor
        max_pooling_factor: Max pooling factor (or None)
        num_heads: Number of attention heads
        
    Returns:
        conformer_input_dim: Adjusted dimension divisible by num_heads
    """
    if max_pooling_factor is None:
        dim_after_pooling = base_feature_dim
    else:
        dim_after_pooling = base_feature_dim // max_pooling_factor
    
    # Find the largest dimension <= dim_after_pooling that's divisible by num_heads
    conformer_input_dim = (dim_after_pooling // num_heads) * num_heads
    
    return conformer_input_dim

"""
Sequence Modeling Alternatives for Partial Spoof Detection
Provides LSTM, Transformer, and CNN architectures as alternatives to Conformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================================
# LSTM-based Sequence Model
# ============================================================================================

class LSTMSequenceModel(nn.Module):
    """
    Bidirectional LSTM for sequence modeling
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for LSTM
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout probability
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(LSTMSequenceModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output dimension is 2*hidden_dim due to bidirectional
        self.output_dim = hidden_dim * 2
        
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            lengths: Sequence lengths (optional, for packing)
            
        Returns:
            output: Tensor of shape (batch_size, seq_len, hidden_dim*2)
            lengths: Updated sequence lengths
        """
        # Pack sequences if lengths provided
        if lengths is not None:
            # Convert lengths to CPU for pack_padded_sequence
            lengths_cpu = lengths.cpu()
            # Sort by length (required for pack_padded_sequence)
            sorted_lengths, sorted_idx = lengths_cpu.sort(descending=True)
            x = x[sorted_idx]
            
            # Pack the sequences
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, sorted_lengths, batch_first=True, enforce_sorted=True
            )
            
            # LSTM forward pass
            packed_output, (hidden, cell) = self.lstm(packed_x)
            
            # Unpack the sequences
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
            
            # Restore original order
            _, unsorted_idx = sorted_idx.sort()
            output = output[unsorted_idx]
        else:
            # Standard LSTM forward pass without packing
            output, (hidden, cell) = self.lstm(x)
        
        # Apply layer normalization and dropout
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output, lengths


# ============================================================================================
# Transformer-based Sequence Model
# ============================================================================================

class TransformerSequenceModel(nn.Module):
    """
    Transformer encoder for sequence modeling
    
    Args:
        input_dim (int): Input feature dimension
        num_heads (int): Number of attention heads
        hidden_dim (int): Feed-forward network dimension
        num_layers (int): Number of transformer layers
        dropout (float): Dropout probability
    """
    def __init__(self, input_dim, num_heads=8, hidden_dim=256, num_layers=4, dropout=0.2):
        super(TransformerSequenceModel, self).__init__()
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN transformer (more stable)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(input_dim)
        )
        
        self.output_dim = input_dim
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            lengths: Sequence lengths (optional, for masking)
            
        Returns:
            output: Tensor of shape (batch_size, seq_len, input_dim)
            lengths: Updated sequence lengths
        """
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask if lengths provided
        src_key_padding_mask = None
        if lengths is not None:
            batch_size, seq_len = x.size(0), x.size(1)
            src_key_padding_mask = torch.arange(seq_len, device=x.device)[None, :] >= lengths[:, None]
        
        # Transformer forward pass
        output = self.transformer_encoder(
            x, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        return output, lengths


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================================
# CNN-based Sequence Model
# ============================================================================================

class CNNSequenceModel(nn.Module):
    """
    CNN for sequence modeling using temporal convolutions
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for CNN
        num_layers (int): Number of convolutional layers
        kernel_size (int): Convolution kernel size
        dropout (float): Dropout probability
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, kernel_size=3, dropout=0.2):
        super(CNNSequenceModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # First layer: input_dim -> hidden_dim
        self.conv_layers.append(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False
            )
        )
        self.norm_layers.append(nn.BatchNorm1d(hidden_dim))
        self.dropout_layers.append(nn.Dropout(dropout))
        
        # Subsequent layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False
                )
            )
            self.norm_layers.append(nn.BatchNorm1d(hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout))
        
        self.output_dim = hidden_dim
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            lengths: Sequence lengths (optional)
            
        Returns:
            output: Tensor of shape (batch_size, seq_len, hidden_dim)
            lengths: Updated sequence lengths
        """
        # Transpose to (batch_size, input_dim, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        for conv, norm, dropout in zip(self.conv_layers, self.norm_layers, self.dropout_layers):
            residual = x if x.size(1) == self.hidden_dim else None
            
            x = conv(x)
            x = norm(x)
            x = F.gelu(x)
            x = dropout(x)
            
            # Residual connection if dimensions match
            if residual is not None:
                x = x + residual
        
        # Transpose back to (batch_size, seq_len, hidden_dim)
        x = x.transpose(1, 2)
        
        return x, lengths


# ============================================================================================
# Temporal Convolutional Network (TCN) - Advanced CNN variant
# ============================================================================================

class TCNSequenceModel(nn.Module):
    """
    Temporal Convolutional Network with dilated convolutions
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for TCN
        num_layers (int): Number of TCN blocks
        kernel_size (int): Convolution kernel size
        dropout (float): Dropout probability
    """
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, kernel_size=3, dropout=0.2):
        super(TCNSequenceModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build TCN blocks with increasing dilation
        self.tcn_blocks = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else hidden_dim
            
            self.tcn_blocks.append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        self.output_dim = hidden_dim
        
    def forward(self, x, lengths=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            lengths: Sequence lengths (optional)
            
        Returns:
            output: Tensor of shape (batch_size, seq_len, hidden_dim)
            lengths: Updated sequence lengths
        """
        # Transpose to (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Apply TCN blocks
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)
        
        # Transpose back to (batch_size, seq_len, hidden_dim)
        x = x.transpose(1, 2)
        
        return x, lengths


class TCNBlock(nn.Module):
    """
    Single TCN block with dilated convolution and residual connection
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TCNBlock, self).__init__()
        
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)
        """
        residual = x
        
        # First convolution
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # Trim to match input length
        out = self.bn1(out)
        out = F.gelu(out)
        out = self.dropout1(out)
        
        # Second convolution
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]  # Trim to match input length
        out = self.bn2(out)
        out = F.gelu(out)
        out = self.dropout2(out)
        
        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        return F.gelu(out + residual)


# ============================================================================================
# Model Factory Function
# ============================================================================================

def create_sequence_model(model_type, input_dim, config):
    """
    Factory function to create sequence models
    
    Args:
        model_type (str): Type of model ('lstm', 'transformer', 'cnn', 'tcn', 'conformer')
        input_dim (int): Input feature dimension
        config (dict): Configuration dictionary with model parameters
        
    Returns:
        model: Instantiated sequence model
    """
    model_type = model_type.lower()
    
    if model_type == 'lstm':
        return LSTMSequenceModel(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.2)
        )
    
    elif model_type == 'transformer':
        return TransformerSequenceModel(
            input_dim=input_dim,
            num_heads=config.get('num_heads', 8),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 4),
            dropout=config.get('dropout', 0.2)
        )
    
    elif model_type == 'cnn':
        return CNNSequenceModel(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 4),
            kernel_size=config.get('kernel_size', 3),
            dropout=config.get('dropout', 0.2)
        )
    
    elif model_type == 'tcn':
        return TCNSequenceModel(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 4),
            kernel_size=config.get('kernel_size', 3),
            dropout=config.get('dropout', 0.2)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Supported types: 'lstm', 'transformer', 'cnn', 'tcn', 'conformer'")


# ============================================================================================
# Example Usage
# ============================================================================================

if __name__ == "__main__":
    # Test all sequence models
    batch_size = 4
    seq_len = 100
    input_dim = 768
    
    x = torch.randn(batch_size, seq_len, input_dim)
    lengths = torch.tensor([100, 90, 80, 70], dtype=torch.int16)
    
    print("Testing Sequence Models:")
    print("=" * 60)
    
    # Test LSTM
    print("\n1. LSTM Model:")
    lstm_model = LSTMSequenceModel(input_dim, hidden_dim=128, num_layers=2)
    lstm_out, _ = lstm_model(x, lengths)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {lstm_out.shape}")
    print(f"   Output dim: {lstm_model.output_dim}")
    
    # Test Transformer
    print("\n2. Transformer Model:")
    transformer_model = TransformerSequenceModel(input_dim, num_heads=8, hidden_dim=256, num_layers=4)
    transformer_out, _ = transformer_model(x, lengths)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {transformer_out.shape}")
    print(f"   Output dim: {transformer_model.output_dim}")
    
    # Test CNN
    print("\n3. CNN Model:")
    cnn_model = CNNSequenceModel(input_dim, hidden_dim=256, num_layers=4)
    cnn_out, _ = cnn_model(x, lengths)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {cnn_out.shape}")
    print(f"   Output dim: {cnn_model.output_dim}")
    
    # Test TCN
    print("\n4. TCN Model:")
    tcn_model = TCNSequenceModel(input_dim, hidden_dim=256, num_layers=4)
    tcn_out, _ = tcn_model(x, lengths)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {tcn_out.shape}")
    print(f"   Output dim: {tcn_model.output_dim}")
    
    print("\n" + "=" * 60)
    print("All models tested successfully!")

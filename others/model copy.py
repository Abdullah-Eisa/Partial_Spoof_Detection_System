import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from utils.utils import *
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import os
import torch.optim as optim
import torchaudio.models as tam

# ============================================================================================
# SAP = SelfWeightedPooling

import torch.nn.init as torch_init

# ============================================================================================
# Pooling Strategy Classes

# class LearnedFeatureProjection(nn.Module):
#     """
#     Attention Pooling: Projects features with learned attention weights.
#     Clean implementation for attention-based feature projection.
    
#     Input: (batch, time, input_dim)  - e.g., (B, T, 768)
#     Output: (batch, output_dim) - projected and pooled features
#     """
#     def __init__(self, input_dim, output_dim):
#         super(LearnedFeatureProjection, self).__init__()
        
#         self.input_dim = input_dim
#         self.output_dim = output_dim
        
#         # Learnable attention matrix: each output feature attends to input features
#         self.attention_weights = nn.Parameter(
#             torch.Tensor(output_dim, input_dim),
#             requires_grad=True
#         )
#         torch_init.xavier_uniform_(self.attention_weights)
    
#     def forward(self, inputs):
#         """
#         Input: (batch, time, input_dim)  - e.g., (B, T, 768)
#         Output: (batch, output_dim) - e.g., (B, 256)
#         """
#         # Compute attention weights (softmax over input dimension)
#         attention = F.softmax(self.attention_weights, dim=1)  # (output_dim, input_dim)
        
#         # Apply weighted projection across time dimension
#         # inputs: (B, T, input_dim)
#         # attention.t(): (input_dim, output_dim)
#         # output: (B, T, output_dim)
#         projected = torch.matmul(inputs, attention.t())
        
#         # Pool across time dimension (mean pooling)
#         output = torch.mean(projected, dim=1)  # (B, output_dim)
        
#         return output


# class AveragePooling(nn.Module):
#     """
#     Average Pooling wrapper for downsampling features across input_dim.
    
#     Input: (batch, time, input_dim)
#     Output: (batch, time, downsampled_input_dim)
#     """
#     def __init__(self, kernel_size, stride):
#         super(AveragePooling, self).__init__()
#         self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
#     def forward(self, x):
#         """
#         Input: (batch, time, input_dim)
#         Output: (batch, time, downsampled_input_dim)
        
#         Pool across the input_dim (feature) dimension, not the time dimension
#         """
#         # Get dimensions
#         batch_size, time_steps, input_dim = x.size()
        
#         # Reshape to (batch*time, 1, input_dim) for AvgPool1d
#         x_reshaped = x.reshape(batch_size * time_steps, 1, input_dim)
        
#         # Apply average pooling on input_dim
#         x_pooled = self.pool(x_reshaped)
        
#         # Get the output dimension after pooling
#         output_dim = x_pooled.size(2)
        
#         # Reshape back to (batch, time, downsampled_input_dim)
#         x = x_pooled.reshape(batch_size, time_steps, output_dim)
        
#         return x


# class StridedConvPooling(nn.Module):
#     """
#     Strided Convolution Pooling for downsampling features across input_dim.
    
#     Input: (batch, time, input_dim)  - e.g., (B, T, 768)
#     Output: (batch, time, out_channels)
#     """
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=3, padding=0):
#         super(StridedConvPooling, self).__init__()
#         self.downsample = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             bias=False
#         )
    
#     def forward(self, x):
#         """
#         Input: (batch, time, input_dim)  - e.g., (B, T, 768)
#         Output: (batch, time, out_channels)
        
#         Apply strided convolution across the input_dim (feature) dimension
#         """
#         batch_size, time_steps, input_dim = x.size()
        
#         # Reshape to (batch*time, 1, input_dim) for Conv1d
#         x_reshaped = x.reshape(batch_size * time_steps, 1, input_dim)
        
#         # Apply strided convolution on input_dim
#         x_conv = self.downsample(x_reshaped)
        
#         # Get the output channels after convolution
#         out_channels = x_conv.size(1)
        
#         # Reshape back to (batch, time, out_channels)
#         x = x_conv.reshape(batch_size, time_steps, out_channels)
        
#         return x




# class LearnedFeatureProjection(nn.Module):
#     """
#     Attention Pooling: Projects features with learned attention weights.
    
#     Input: (batch, time, input_dim)  - e.g., (B, T, 768)
#     Output: (batch, time, output_dim) - e.g., (B, T, 256)
#     """
#     def __init__(self, input_dim, output_dim):
#         super(LearnedFeatureProjection, self).__init__()
        
#         self.input_dim = input_dim
#         self.output_dim = output_dim
        
#         # Linear projection layer instead of attention weights
#         self.projection = nn.Linear(input_dim, output_dim)
    
#     def forward(self, inputs):
#         """
#         Input: (batch, time, input_dim)  - e.g., (B, T, 768)
#         Output: (batch, time, output_dim) - e.g., (B, T, 256)
#         """
#         # Apply linear projection across feature dimension
#         # inputs: (B, T, input_dim)
#         # output: (B, T, output_dim)
#         output = self.projection(inputs)
        
#         return output


# class AveragePooling(nn.Module):
#     """
#     Average Pooling for downsampling features across input_dim.
    
#     Input: (batch, time, input_dim)
#     Output: (batch, time, downsampled_input_dim)
#     """
#     def __init__(self, kernel_size, stride):
#         super(AveragePooling, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    
#     def forward(self, x):
#         """
#         Input: (batch, time, input_dim)
#         Output: (batch, time, downsampled_input_dim)
        
#         Pool across the input_dim (feature) dimension, keeping time dimension intact
#         """
#         # Get dimensions
#         batch_size, time_steps, input_dim = x.size()
        
#         # Reshape to (batch*time, 1, input_dim) to pool along feature dimension
#         # Each timestep is treated independently
#         x_reshaped = x.reshape(batch_size * time_steps, 1, input_dim)
        
#         # Apply average pooling on feature dimension
#         x_pooled = self.pool(x_reshaped)  # (batch*time, 1, downsampled_input_dim)
        
#         # Get the output dimension after pooling
#         output_dim = x_pooled.size(2)
        
#         # Reshape back to (batch, time, downsampled_input_dim)
#         x = x_pooled.reshape(batch_size, time_steps, output_dim)
        
#         return x


# class StridedConvPooling(nn.Module):
#     """
#     Strided Convolution Pooling for downsampling features across input_dim.
    
#     Input: (batch, time, input_dim)  - e.g., (B, T, 768)
#     Output: (batch, time, out_channels)
#     """
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=3, padding=0):
#         super(StridedConvPooling, self).__init__()
#         self.downsample = nn.Conv1d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=padding,
#             bias=False
#         )
    
#     def forward(self, x):
#         """
#         Input: (batch, time, input_dim)  - e.g., (B, T, 768)
#         Output: (batch, time, out_channels)
        
#         Apply strided convolution across the input_dim (feature) dimension,
#         keeping time dimension intact
#         """
#         batch_size, time_steps, input_dim = x.size()
        
#         # Reshape to (batch*time, in_channels, 1) to apply Conv1d on feature dimension
#         # Each timestep is treated independently
#         x_reshaped = x.reshape(batch_size * time_steps, input_dim, 1)
        
#         # Apply strided convolution on feature dimension
#         x_conv = self.downsample(x_reshaped)  # (batch*time, out_channels, 1)
        
#         # Get the output channels
#         out_channels = x_conv.size(1)
        
#         # Reshape back to (batch, time, out_channels)
#         x = x_conv.reshape(batch_size, time_steps, out_channels)
        
#         return x


# class PoolingFactory:
#     """
#     Factory class for creating pooling strategies.
#     """
#     @staticmethod
#     def create_pooling(strategy, input_dim, config):
#         """
#         Create a pooling module based on the specified strategy.
        
#         Args:
#             strategy: str, pooling strategy name
#             input_dim: int, input feature dimension
#             config: dict, configuration dictionary containing pooling parameters
        
#         Returns:
#             tuple: (pooling_module, output_dim)
#         """
#         strategy = strategy.lower()
        
#         if strategy == "average":
#             kernel_size = config['model']['average_pooling']['kernel_size']
#             stride = config['model']['average_pooling']['stride']
#             pooling = AveragePooling(kernel_size=kernel_size, stride=stride)
#             # Calculate output dimension after average pooling on input_dim
#             output_dim = (input_dim - kernel_size) // stride + 1
#             return pooling, output_dim
        
#         elif strategy == "attention":
#             output_dim = config['model']['attention_pooling']['output_dim']
#             pooling = LearnedFeatureProjection(input_dim=input_dim, output_dim=output_dim)
#             # Note: attention pooling also applies global pooling (mean), so final output is just output_dim
#             return pooling, output_dim
        
#         elif strategy == "strided_conv":
#             out_channels = config['model']['strided_conv_pooling']['out_channels']
#             kernel_size = config['model']['strided_conv_pooling']['kernel_size']
#             stride = config['model']['strided_conv_pooling']['stride']
#             padding = config['model']['strided_conv_pooling']['padding']
#             pooling = StridedConvPooling(
#                 in_channels=input_dim,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding
#             )
#             output_dim = out_channels
#             return pooling, output_dim
        
#         elif strategy == "self_weighted":
#             # SelfWeightedPooling will be handled separately in the model
#             return None, input_dim
        
#         elif strategy == "max":
#             # Max pooling is handled separately in the model
#             return None, input_dim
        
#         else:
#             raise ValueError(f"Unknown pooling strategy: {strategy}")






import torch
import torch.nn as nn


class LearnedFeatureProjection(nn.Module):
    """
    Attention Pooling: Projects features with learned attention weights.
    
    Input: (batch, time, input_dim)  - e.g., (B, T, 768)
    Output: (batch, time, output_dim) - e.g., (B, T, 256)
    
    Downsamples FEATURE DIMENSION, keeps time dimension intact.
    """
    def __init__(self, input_dim, output_dim):
        super(LearnedFeatureProjection, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Linear projection layer to reduce feature dimension
        self.projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, inputs):
        """
        Input: (batch, time, input_dim)  - e.g., (B, T, 768)
        Output: (batch, time, output_dim) - e.g., (B, T, 256)
        """
        # Apply linear projection across feature dimension
        # inputs: (B, T, input_dim)
        # output: (B, T, output_dim)
        output = self.projection(inputs)
        
        return output


class MaxPooling(nn.Module):
    """
    Max Pooling for downsampling FEATURE DIMENSION (not time dimension).
    
    Input: (batch, time, input_dim)
    Output: (batch, time, downsampled_input_dim)
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=self.stride, padding=padding)
    
    def forward(self, x):
        """
        Input: (batch, time, input_dim)
        Output: (batch, time, downsampled_input_dim)
        
        Pool across the input_dim (feature) dimension, keeping time dimension intact.
        """
        # Get dimensions
        batch_size, time_steps, input_dim = x.size()
        
        # Transpose to (batch, input_dim, time) for pooling
        x = x.transpose(1, 2)  # (B, input_dim, T)
        
        # Reshape to treat each time step's features independently
        # We want to pool along feature dimension while preserving time
        # Approach: reshape to (batch*time, 1, input_dim)
        x_reshaped = x.transpose(1, 2).reshape(batch_size * time_steps, 1, input_dim)
        
        # Apply max pooling on feature dimension
        x_pooled = self.pool(x_reshaped)  # (batch*time, 1, downsampled_input_dim)
        
        # Get the output dimension after pooling
        output_dim = x_pooled.size(2)
        
        # Reshape back to (batch, time, downsampled_input_dim)
        x = x_pooled.reshape(batch_size, time_steps, output_dim)
        
        return x


class AveragePooling(nn.Module):
    """
    Average Pooling for downsampling FEATURE DIMENSION (not time dimension).
    
    Input: (batch, time, input_dim)
    Output: (batch, time, downsampled_input_dim)
    """
    def __init__(self, kernel_size, stride=None, padding=0):
        super(AveragePooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=self.stride, padding=padding)
    
    def forward(self, x):
        """
        Input: (batch, time, input_dim)
        Output: (batch, time, downsampled_input_dim)
        
        Pool across the input_dim (feature) dimension, keeping time dimension intact.
        """
        # Get dimensions
        batch_size, time_steps, input_dim = x.size()
        
        # Reshape to (batch*time, 1, input_dim) to pool along feature dimension
        # Each timestep is treated independently
        x_reshaped = x.reshape(batch_size * time_steps, 1, input_dim)
        
        # Apply average pooling on feature dimension
        x_pooled = self.pool(x_reshaped)  # (batch*time, 1, downsampled_input_dim)
        
        # Get the output dimension after pooling
        output_dim = x_pooled.size(2)
        
        # Reshape back to (batch, time, downsampled_input_dim)
        x = x_pooled.reshape(batch_size, time_steps, output_dim)
        
        return x


class StridedConvPooling(nn.Module):
    """
    Strided Convolution Pooling for downsampling FEATURE DIMENSION (not time dimension).
    
    Input: (batch, time, input_dim)  - e.g., (B, T, 768)
    Output: (batch, time, output_dim)
    """
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=None, padding=0):
        super(StridedConvPooling, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.stride = stride if stride is not None else kernel_size
        
        # Conv1d that operates on feature dimension
        self.downsample = nn.Conv1d(
            in_channels=1,  # Each time step treated independently
            out_channels=1,  # Keep same structure
            kernel_size=kernel_size,
            stride=self.stride,
            padding=padding,
            bias=True
        )
        
        # Calculate expected output dimension after convolution
        expected_output_dim = (input_dim + 2 * padding - kernel_size) // self.stride + 1
        
        # Add a linear projection to get exact output_dim
        self.projection = nn.Linear(expected_output_dim, output_dim)
    
    def forward(self, x):
        """
        Input: (batch, time, input_dim)  - e.g., (B, T, 768)
        Output: (batch, time, output_dim)
        
        Apply strided convolution across the input_dim (feature) dimension,
        keeping time dimension intact.
        """
        batch_size, time_steps, input_dim = x.size()
        
        # Reshape to (batch*time, 1, input_dim) to apply Conv1d on feature dimension
        # Each timestep is treated independently
        x_reshaped = x.reshape(batch_size * time_steps, 1, input_dim)
        
        # Apply strided convolution on feature dimension
        x_conv = self.downsample(x_reshaped)  # (batch*time, 1, conv_output_dim)
        
        # Remove channel dimension and get features
        x_conv = x_conv.squeeze(1)  # (batch*time, conv_output_dim)
        
        # Apply linear projection to get exact output_dim
        x_projected = self.projection(x_conv)  # (batch*time, output_dim)
        
        # Reshape back to (batch, time, output_dim)
        x = x_projected.reshape(batch_size, time_steps, self.output_dim)
        
        return x


class PoolingFactory:
    """
    Factory class for creating pooling strategies.
    All strategies downsample FEATURE DIMENSION, not time dimension.
    """
    @staticmethod
    def create_pooling(strategy, input_dim, config):
        """
        Create a pooling module based on the specified strategy.
        
        Args:
            strategy: str, pooling strategy name
            input_dim: int, input feature dimension (e.g., 768)
            config: dict, configuration dictionary containing pooling parameters
        
        Returns:
            tuple: (pooling_module, output_dim)
                   output_dim is the downsampled feature dimension
        """
        strategy = strategy.lower()
        
        if strategy == "max":
            kernel_size = config['model']['max_pooling']['kernel_size']
            stride = config['model']['max_pooling'].get('stride', kernel_size)
            padding = config['model']['max_pooling'].get('padding', 0)
            pooling = MaxPooling(kernel_size=kernel_size, stride=stride, padding=padding)
            # Calculate output dimension after max pooling on feature dimension
            output_dim = (input_dim + 2 * padding - kernel_size) // stride + 1
            return pooling, output_dim
        
        elif strategy == "average":
            kernel_size = config['model']['average_pooling']['kernel_size']
            stride = config['model']['average_pooling'].get('stride', kernel_size)
            padding = config['model']['average_pooling'].get('padding', 0)
            pooling = AveragePooling(kernel_size=kernel_size, stride=stride, padding=padding)
            # Calculate output dimension after average pooling on feature dimension
            output_dim = (input_dim + 2 * padding - kernel_size) // stride + 1
            return pooling, output_dim
        
        elif strategy == "attention":
            output_dim = config['model']['attention_pooling']['output_dim']
            pooling = LearnedFeatureProjection(input_dim=input_dim, output_dim=output_dim)
            return pooling, output_dim
        
        elif strategy == "strided_conv":
            output_dim = config['model']['strided_conv_pooling']['output_dim']
            kernel_size = config['model']['strided_conv_pooling']['kernel_size']
            stride = config['model']['strided_conv_pooling'].get('stride', kernel_size)
            padding = config['model']['strided_conv_pooling'].get('padding', 0)
            pooling = StridedConvPooling(
                input_dim=input_dim,
                output_dim=output_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            return pooling, output_dim
        
        elif strategy == "self_weighted":
            # SelfWeightedPooling will be handled separately in the model
            # This pools across TIME dimension, not feature dimension
            return None, input_dim
        
        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")










# ============================================================================================
# code adapted from: https://github.com/nii-yamagishilab/PartialSpoof/blob/847347aaec6f65c3c6d2f17c63515b826b94feb3/project-NN-Pytorch-scripts.202102/sandbox/block_nn.py#L709
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
# code adapted from: https://pytorch.org/torchtune/stable/_modules/torchtune/modules/position_embeddings.html

from typing import Optional

import torch
from torch import nn

class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        return x_out.type_as(x)



# ============================================================================================
# ============================================================================================
# binary classification model with Rotary Positional Encoding and max pooling after feature extractor

# class BinarySpoofingClassificationModel(nn.Module):
#     def __init__(self, feature_dim, num_heads, hidden_dim, max_dropout=0.2, depthwise_conv_kernel_size=31, conformer_layers=1, max_pooling_factor=3):
#         super(BinarySpoofingClassificationModel, self).__init__()

#         self.max_pooling_factor = max_pooling_factor
#         self.feature_dim = feature_dim
#         self.num_heads = num_heads
#         self.max_dropout=max_dropout

#         if self.max_pooling_factor is not None:
#             self.max_pooling = nn.MaxPool1d(kernel_size=self.max_pooling_factor, stride=self.max_pooling_factor)
#             self.feature_dim=feature_dim//self.max_pooling_factor
#         else:
#             self.max_pooling = None
        
#         print(f"self.feature_dim= {self.feature_dim} , self.max_pooling= {self.max_pooling}")
#         # Define the Conformer model from torchaudio
#         self.conformer = tam.Conformer(
#             input_dim=self.feature_dim,
#             num_heads=self.num_heads,
#             ffn_dim=hidden_dim,  # Feed-forward network dimension (for consistency)
#             num_layers=conformer_layers,  # Example, adjust as needed
#             depthwise_conv_kernel_size=depthwise_conv_kernel_size,  # Set the kernel size for depthwise convolution
#             dropout=0.2, # Dropout initialized with 0.2 for conformer block
#             use_group_norm= False, 
#             convolution_first= False
#         )
        
#         # Global pooling layer (SelfWeightedPooling)
#         self.pooling = SelfWeightedPooling(self.feature_dim , mean_only=True)  # Pool across sequence dimension
        
#         # Add a feedforward block for feature refinement before classification
#         self.fc_refinement = nn.Sequential(
#             nn.Linear(self.feature_dim, hidden_dim),  # Refined hidden dimension for classification
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(0.2),  # Dropout for regularization

#             nn.Linear(hidden_dim, hidden_dim//2),  # Refined hidden dimension for classification
#             nn.LayerNorm(hidden_dim//2),
#             nn.GELU(),
#             nn.Dropout(0.2),  # Dropout for regularization

#             nn.Linear(hidden_dim//2, hidden_dim//4),  # Refined hidden dimension for classification
#             nn.LayerNorm(hidden_dim//4),
#             nn.GELU(),
#             nn.Dropout(0.2),  # Dropout for regularization

#             nn.Linear(hidden_dim//4, 1),  # Final output layer
#             # nn.Sigmoid(),
#         )


#         self.apply(self.initialize_weights)


class BinarySpoofingClassificationModel(nn.Module):
    def __init__(self, feature_dim, num_heads, hidden_dim, max_dropout=0.2, 
                 depthwise_conv_kernel_size=31, conformer_layers=1, max_pooling_factor=3,
                 use_max_pooling=True, pooling_strategy="self_weighted", config=None):
        super(BinarySpoofingClassificationModel, self).__init__()

        self.max_pooling_factor = max_pooling_factor
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.max_dropout = max_dropout
        self.use_max_pooling = use_max_pooling
        self.pooling_strategy = pooling_strategy.lower()
        self.config = config

        # Initialize base conformer input dimension
        self.conformer_input_dim = feature_dim
        
        # Apply downsampling strategies
        if self.pooling_strategy == "average":
            # Average pooling
            kernel_size = config['model']['average_pooling']['kernel_size'] if config else 3
            stride = config['model']['average_pooling']['stride'] if config else 3
            self.downsample = AveragePooling(kernel_size=kernel_size, stride=stride)
            self.conformer_input_dim = feature_dim  # Output dim same as input
            
        elif self.pooling_strategy == "attention":
            # Attention pooling (LearnedFeatureProjection)
            output_dim = config['model']['attention_pooling']['output_dim'] if config else 256
            self.downsample = LearnedFeatureProjection(input_dim=feature_dim, output_dim=output_dim)
            self.conformer_input_dim = output_dim
            
        elif self.pooling_strategy == "strided_conv":
            # Strided convolution pooling
            out_channels = config['model']['strided_conv_pooling']['out_channels'] if config else 256
            kernel_size = config['model']['strided_conv_pooling']['kernel_size'] if config else 3
            stride = config['model']['strided_conv_pooling']['stride'] if config else 3
            padding = config['model']['strided_conv_pooling']['padding'] if config else 0
            self.downsample = StridedConvPooling(
                in_channels=feature_dim,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            self.conformer_input_dim = out_channels
            
        elif self.pooling_strategy == "max":
            # Max pooling
            if self.use_max_pooling and self.max_pooling_factor is not None:
                self.downsample = nn.MaxPool1d(kernel_size=self.max_pooling_factor, stride=self.max_pooling_factor)
                self.conformer_input_dim = feature_dim // self.max_pooling_factor
            else:
                self.downsample = None
                
        elif self.pooling_strategy == "self_weighted":
            # Self-weighted pooling (no downsampling, pooling applied later)
            self.downsample = None
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        # Ensure dimension is divisible by num_heads
        self.conformer_input_dim = (self.conformer_input_dim // num_heads) * num_heads
        
        print(f"Pooling Strategy: {self.pooling_strategy}")
        print(f"Feature Dim: {self.feature_dim}")
        print(f"Conformer Input Dim: {self.conformer_input_dim}")
        print(f"Num Heads: {self.num_heads}")
        
        # Verify dimension is divisible by num_heads
        assert self.conformer_input_dim % num_heads == 0, \
            f"conformer_input_dim ({self.conformer_input_dim}) must be divisible by num_heads ({num_heads})"
        
        # Define the Conformer model from torchaudio
        self.conformer = tam.Conformer(
            input_dim=self.conformer_input_dim,
            num_heads=self.num_heads,
            ffn_dim=hidden_dim,
            num_layers=conformer_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=0.2,
            use_group_norm=False, 
            convolution_first=False
        )
        
        # Global pooling layer (SelfWeightedPooling)
        self.pooling = SelfWeightedPooling(self.conformer_input_dim, mean_only=True)
        
        # Add a feedforward block for feature refinement before classification
        self.fc_refinement = nn.Sequential(
            nn.Linear(self.conformer_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.LayerNorm(hidden_dim//4),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim//4, 1),
        )

        self.apply(self.initialize_weights)

    # Custom initialization for He and Xavier
    def initialize_weights(self, m, bias_value=0.005):
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


    # def forward(self, x, lengths,dropout_prob):
    #     if self.max_pooling is not None:
    #         x = self.max_pooling(x)  # Apply max pooling

    #     # Apply Conformer model
    #     x, _ = self.conformer(x, lengths)  # The second returned value is the sequence lengths
        
    #     # Apply global pooling across the sequence dimension (SelfWeightedPooling)
    #     x = self.pooling(x)  # Now x is (batch_size, hidden_dim, 1)

    #     # Update the dropout probability dynamically
    #     self.fc_refinement[3].p = dropout_prob  # Update the dropout layer's probability
    #     self.fc_refinement[7].p = dropout_prob  # Update the dropout layer's probability
    #     self.fc_refinement[11].p = dropout_prob  # Update the dropout layer's probability

    #     # Refine features before classification using the fc_refinement block
    #     utt_score = self.fc_refinement(x)
    #     return utt_score # Return the classification output
        
    def forward(self, x, lengths, dropout_prob):
        """
        Forward pass through the model with support for different pooling strategies.
        
        Args:
            x: Input features (batch, time, feature_dim)
            lengths: Lengths of sequences
            dropout_prob: Dropout probability for the epoch
        
        Returns:
            Output scores (batch, 1)
        """
        # Apply downsampling strategy
        if self.pooling_strategy == "average":
            # Average pooling
            x = self.downsample(x)
            # Update lengths after downsampling
            kernel_size = self.config['model']['average_pooling']['kernel_size'] if self.config else 3
            stride = self.config['model']['average_pooling']['stride'] if self.config else 3
            lengths = ((lengths - kernel_size) // stride + 1).clamp(min=1)
            
        elif self.pooling_strategy == "attention":
            # Attention pooling (LearnedFeatureProjection) - returns pooled features
            x = self.downsample(x)  # Output is already pooled globally: (batch, output_dim)
            # Need to reshape for conformer input if it processes sequences
            # If attention pooling is used, we get (batch, output_dim) directly
            # So we need to expand it to (batch, 1, output_dim) for conformer
            x = x.unsqueeze(1)  # (batch, 1, output_dim)
            lengths = torch.ones_like(lengths)  # All sequences have length 1 after global pooling
            
        elif self.pooling_strategy == "strided_conv":
            # Strided convolution pooling
            x = self.downsample(x)  # (batch, time, out_channels)
            # Update lengths after strided convolution
            kernel_size = self.config['model']['strided_conv_pooling']['kernel_size'] if self.config else 3
            stride = self.config['model']['strided_conv_pooling']['stride'] if self.config else 3
            padding = self.config['model']['strided_conv_pooling']['padding'] if self.config else 0
            lengths = ((lengths + 2 * padding - kernel_size) // stride + 1).clamp(min=1)
            
        elif self.pooling_strategy == "max":
            # Max pooling
            if self.downsample is not None:
                x = self.downsample(x)
                # Update lengths after max pooling
                lengths = ((lengths + self.max_pooling_factor - 1) // self.max_pooling_factor).clamp(min=1)
        
        # Apply Conformer model
        x, _ = self.conformer(x, lengths)
        
        # Apply global pooling across the sequence dimension (SelfWeightedPooling)
        x = self.pooling(x)

        # Update the dropout probability dynamically
        self.fc_refinement[3].p = dropout_prob
        self.fc_refinement[7].p = dropout_prob
        self.fc_refinement[11].p = dropout_prob

        # Refine features before classification
        utt_score = self.fc_refinement(x)
        return utt_score


    def adjust_dropout(self, epoch, total_epochs):
        # Cosine annealing for dropout probability
        return self.max_dropout * (1 + math.cos(math.pi * epoch / total_epochs)) / 2



# ===========================================================================================================================
# ===========================================================================================================================

# def initialize_models(ssl_ckpt_path, save_feature_extractor=False,
#                       feature_dim=768, num_heads=8, hidden_dim=128, max_dropout=0.2, depthwise_conv_kernel_size=31, conformer_layers=1, max_pooling_factor=3, 
#                       LEARNING_RATE=0.0001,DEVICE='cpu'):
#     """Initialize the model, feature extractor, and optimizer"""
#     # Initialize feature extractor
#     if os.path.exists(ssl_ckpt_path):
#         feature_extractor = torch.hub.load('s3prl/s3prl', 'wav2vec2', model_path=ssl_ckpt_path).to(DEVICE)
#     else:
#         ssl_ckpt_path = os.path.join(os.getcwd(), 'models/w2v_large_lv_fsh_swbd_cv.pt')
#         feature_extractor = torch.hub.load('s3prl/s3prl', 'wav2vec2', model_path=ssl_ckpt_path).to(DEVICE)

#     # Initialize Binary Spoofing Classification Model
#     PS_Model = BinarySpoofingClassificationModel(feature_dim, num_heads, hidden_dim, max_dropout, depthwise_conv_kernel_size, conformer_layers, max_pooling_factor).to(DEVICE)

#     # Freeze feature extractor if necessary
#     if save_feature_extractor:
#         for name, param in feature_extractor.named_parameters():
#             if 'final_proj' not in name:
#                 param.requires_grad = False
#             else:
#                 param.requires_grad = True
    
#     # Optimizer setup
#     optimizer = optim.AdamW(
#         [{'params': feature_extractor.parameters(), 'lr': 0.00005},
#          {'params': PS_Model.parameters()}], 
#         lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8) if save_feature_extractor else optim.AdamW(PS_Model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)

#     return PS_Model, feature_extractor, optimizer





# def initialize_models(config, save_feature_extractor=False, LEARNING_RATE=0.0001, DEVICE='cpu'):
#     """Initialize the model, feature extractor, and optimizer"""
#     from feature_extractors import FeatureExtractorFactory, get_feature_dim_from_config
    
#     # Create feature extractor based on config
#     feature_extractor = FeatureExtractorFactory.create_extractor(config, DEVICE)
    
#     # Get feature dimension from config
#     feature_dim = get_feature_dim_from_config(config)
    
#     # Initialize Binary Spoofing Classification Model
#     PS_Model = BinarySpoofingClassificationModel(
#         feature_dim=feature_dim,
#         num_heads=config['model']['num_heads'],
#         hidden_dim=config['model']['hidden_dim'],
#         max_dropout=config['model']['max_dropout'],
#         depthwise_conv_kernel_size=config['model']['depthwise_conv_kernel_size'],
#         conformer_layers=config['model']['conformer_layers'],
#         max_pooling_factor=config['model']['max_pooling_factor']
#     ).to(DEVICE)

#     # Freeze feature extractor if necessary
#     if save_feature_extractor and hasattr(feature_extractor, 'model'):
#         for name, param in feature_extractor.model.named_parameters():
#             if 'final_proj' not in name:
#                 param.requires_grad = False
#             else:
#                 param.requires_grad = True
    
#     # Optimizer setup
#     if save_feature_extractor and hasattr(feature_extractor, 'parameters'):
#         optimizer = optim.AdamW(
#             [{'params': feature_extractor.parameters(), 'lr': 0.00005},
#              {'params': PS_Model.parameters()}], 
#             lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
#     else:
#         optimizer = optim.AdamW(
#             PS_Model.parameters(), 
#             lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)

#     return PS_Model, feature_extractor, optimizer


def initialize_models(config, save_feature_extractor=False, LEARNING_RATE=0.0001, DEVICE='cpu'):
    """Initialize the model, feature extractor, and optimizer with pooling strategy support"""
    from feature_extractors import FeatureExtractorFactory, get_feature_dim_from_config, calculate_conformer_input_dim
    
    # Create feature extractor based on config
    feature_extractor = FeatureExtractorFactory.create_extractor(config, DEVICE)
    
    # Get base feature dimension from config
    base_feature_dim = feature_extractor.get_feature_dim()
    
    # Get pooling strategy from config
    pooling_strategy = config['model'].get('pooling_strategy', 'self_weighted')
    
    print(f"Feature Extractor Type: {config['feature_extractor']['type']}")
    print(f"Base Feature Dim: {base_feature_dim}")
    print(f"Pooling Strategy: {pooling_strategy}")
    print(f"feature_extractor: {feature_extractor}")
    
    # Initialize Binary Spoofing Classification Model
    PS_Model = BinarySpoofingClassificationModel(
        feature_dim=base_feature_dim,
        num_heads=config['model']['num_heads'],
        hidden_dim=config['model']['hidden_dim'],
        max_dropout=config['model']['max_dropout'],
        depthwise_conv_kernel_size=config['model']['depthwise_conv_kernel_size'],
        conformer_layers=config['model']['conformer_layers'],
        max_pooling_factor=config['model'].get('max_pooling_factor'),
        use_max_pooling=config['model'].get('use_max_pooling', True),
        pooling_strategy=pooling_strategy,
        config=config
    ).to(DEVICE)

    # Freeze feature extractor if necessary
    if save_feature_extractor and hasattr(feature_extractor, 'model'):
        for name, param in feature_extractor.model.named_parameters():
            if 'final_proj' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    # Optimizer setup
    if save_feature_extractor and hasattr(feature_extractor, 'parameters'):
        optimizer = optim.AdamW(
            [{'params': feature_extractor.parameters(), 'lr': 0.00005},
             {'params': PS_Model.parameters()}], 
            lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = optim.AdamW(
            PS_Model.parameters(), 
            lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)

    return PS_Model, feature_extractor, optimizer




def forward_pass(model, features, lengths, dropout_prob):
    """Forward pass through the model"""
    return model(features, lengths, dropout_prob)

def backward_and_optimize(model, loss, optimizer, max_grad_norm):
    """Backward pass and optimizer step"""
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()




def initialize_loss_function():
    """Initialize the loss function (BCE with logits)"""
    return nn.BCEWithLogitsLoss()

def adjust_dropout_prob(model, epoch, NUM_EPOCHS):
    """Adjust dropout rate dynamically during training"""
    return model.adjust_dropout(epoch, NUM_EPOCHS)



def initialize_lr_scheduler(optimizer):
    """Initialize the learning rate scheduler"""
    factor = 4/5
    patience = 5   # Number of epochs with no improvement after which learning rate will be reduced
    threshold=0.009
    min_lr = 0.00001

    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, threshold=threshold, min_lr=min_lr)
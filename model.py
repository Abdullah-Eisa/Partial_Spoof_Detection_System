import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from utils import *
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import os
import torch.optim as optim
import torchaudio.models as tam

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
# ============================================================================================

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
# ============================================================================================
# ============================================================================================


# binary classification model  max pooling after feature extractor
class BinarySpoofingClassificationModel(nn.Module):
    def __init__(self, feature_dim, num_heads, hidden_dim, max_dropout=0.2, depthwise_conv_kernel_size=31, conformer_layers=1, max_pooling_factor=3):
        super(BinarySpoofingClassificationModel, self).__init__()

        self.max_pooling_factor = max_pooling_factor
        self.feature_dim = feature_dim
        self.num_heads = num_heads
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
            num_heads=self.num_heads,
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
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),  # Dropout for regularization

            nn.Linear(hidden_dim, hidden_dim//2),  # Refined hidden dimension for classification
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.2),  # Dropout for regularization

            nn.Linear(hidden_dim//2, hidden_dim//4),  # Refined hidden dimension for classification
            nn.LayerNorm(hidden_dim//4),
            nn.GELU(),
            nn.Dropout(0.2),  # Dropout for regularization

            nn.Linear(hidden_dim//4, 1),  # Final output layer
            # nn.Sigmoid(),
            # nn.GELU(),
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
#             dropout=0.2,
#             use_group_norm= False, 
#             convolution_first= False
#         )
        
#         # Global pooling layer (SelfWeightedPooling)
#         self.pooling = SelfWeightedPooling(self.feature_dim , mean_only=True)  # Pool across sequence dimension
        

#         # Add Rotary Positional Encoding
#         # self.rotary_pos_enc = RotaryPositionalEmbeddings(dim=self.feature_dim)
#         self.rotary_pos_enc = RotaryPositionalEmbeddings(dim=self.feature_dim//self.num_heads)

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
#             # nn.GELU(),
#         )


#         self.apply(self.initialize_weights)

#     # Custom initialization for He and Xavier
#     def initialize_weights(self, m, bias_value=0.005):
#         if isinstance(m, nn.Linear):  # For Linear layers
#             # We do not directly check activation here, since it's separate
#             if isinstance(m, nn.Linear):
#                 if hasattr(m, 'activation') and isinstance(m.activation, nn.ReLU):
#                     # He (Kaiming) initialization for ReLU/GELU layers
#                     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 elif hasattr(m, 'activation') and isinstance(m.activation, nn.GELU):
#                     # He (Kaiming) initialization for GELU layers
#                     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 elif hasattr(m, 'activation') and isinstance(m.activation, (nn.Tanh, nn.Sigmoid)):
#                     # Xavier (Glorot) initialization for tanh/sigmoid layers
#                     nn.init.xavier_normal_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, bias_value)

#         elif isinstance(m, nn.Conv1d):  # For Conv1d layers (typically used in Conformer)
#             if hasattr(m, 'activation') and isinstance(m.activation, nn.ReLU):
#                 # He (Kaiming) initialization for Conv1d with ReLU/GELU
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif hasattr(m, 'activation') and isinstance(m.activation, (nn.Tanh, nn.Sigmoid)):
#                 # Xavier (Glorot) initialization for Conv1d with tanh/sigmoid
#                 nn.init.xavier_normal_(m.weight)
#             if m.bias is not None:
#                 nn.init.constant_(m.bias, bias_value)


#     def forward(self, x, lengths,dropout_prob):
#         # print(f" x size before conformer = {x.size()}")
#         if self.max_pooling is not None:
#             x = self.max_pooling(x)  # Apply max pooling

#         # print(f" x size after max pooling = {x.size()}")
#         # Apply Rotary Positional Encoding to the input
#         # x = self.rotary_pos_enc(x)  # Add Rotary Positional Encoding

#         # Generate the input positions (assuming the sequence length is consistent across the batch)
#         # batch_size, seq_len, _ = x.size()
#         # positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

#         # Apply Rotary Positional Encoding to the input
#         # x = self.rotary_pos_enc(x, input_pos=positions)  # Add Rotary Positional Encoding
#         x = x.view(x.size(0), x.size(1), self.num_heads, self.feature_dim//self.num_heads)

#         x = self.rotary_pos_enc(x)  # Add Rotary Positional Encoding
#         # x=x.flatten(2) # (batch_size, seq_len, hidden_dim)
#         x = x.view(x.size(0), x.size(1),-1) # (batch_size, seq_len, hidden_dim)

#         # Apply Conformer model
#         x, _ = self.conformer(x, lengths)  # The second returned value is the sequence lengths
#         # print(f" x size after conformer = {x.size()}")
        
#         # Apply global pooling across the sequence dimension (SelfWeightedPooling)
#         x = self.pooling(x)  # Now x is (batch_size, hidden_dim, 1)
#         # print(f" x size after pooling = {x.size()}")

#         # Update the dropout probability dynamically
#         self.fc_refinement[3].p = dropout_prob  # Update the dropout layer's probability
#         self.fc_refinement[7].p = dropout_prob  # Update the dropout layer's probability
#         self.fc_refinement[11].p = dropout_prob  # Update the dropout layer's probability

#         # Refine features before classification using the fc_refinement block
#         utt_score = self.fc_refinement(x)
#         return utt_score # Return the classification output

#     def adjust_dropout(self, epoch, total_epochs):
#         # Cosine annealing for dropout probability
#         return self.max_dropout * (1 + math.cos(math.pi * epoch / total_epochs)) / 2


# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================
# ===========================================================================================================================


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
        [{'params': feature_extractor.parameters(), 'lr': 0.00005},
         {'params': PS_Model.parameters()}], 
        lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8) if save_feature_extractor else optim.AdamW(PS_Model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)

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
    factor = 3/4
    patience = 5   # Number of epochs with no improvement after which learning rate will be reduced
    threshold=0.009
    min_lr = 0.000005

    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, threshold=threshold, min_lr=min_lr)
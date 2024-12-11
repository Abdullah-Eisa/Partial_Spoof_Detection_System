
import torch
# from transformers import Wav2Vec2Processor, 
import torch.nn.functional as F

import torch
import torch.nn as nn


from gmlp import GMLPBlock
from utils import *




# gMLP

# class MyModel(nn.Module):
#     def __init__(self,  d_model=768, d_ffn=256, seq_len=2001, gmlp_layers = 1, batch_first=True,flag_pool = 'ap'):
#         super(MyModel, self).__init__()


#         self.batch_first = batch_first
#         self.flag_pool = flag_pool
#         self.d_model = d_model
#         self.SelfWeightedPooling=SelfWeightedPooling(self.d_model, mean_only=True)

#         if(d_ffn > 0):
#             pass
#         elif(d_ffn < 0):
#             #if emb_dim <0, we will reduce dim by emb_dim. like -2 will be dim/2
#             d_ffn = int(d_model / abs(d_ffn))

#         layers = []
#         for i in range(gmlp_layers):
#             layers.append(GMLPBlock(d_model, d_ffn, seq_len))
#         self.layers = nn.Sequential(*layers)

#         self.fc1 = nn.Linear(d_model, d_ffn, bias=False)

#         # Additional linear layers for classification
#         self.fc2 = nn.Linear(d_ffn, 33)


#     def forward(self, x):

#         x = self.layers(x)

#         #pool for utt
#         if(self.flag_pool == "ap"): #average pooling
#             x = x.mean(dim=1)
#             #nn.AdaptiveAvgPool2d(1)(x)
#         elif(self.flag_pool == "sap"):
#             x = self.SelfWeightedPooling(x)
#         else:
#             pass

#         x = self.fc1(x)
#         x = F.gelu(x)
        
#         x = self.fc2(x)
#         x = torch.sigmoid(x)

#         print(f'in SimpleNN model gmlp with fc output of type {type(x)}  with size {x.size()}')

#         return x



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils.rnn import pad_sequence
# from gmlp import GMLPBlock  # Ensure that you have this import if needed
# from utils import *  # Ensure that you have this import if needed

# class SelfAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(SelfAttention, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads
        
#         # Define linear transformations for query, key, and value
#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)
        
#         # Dropout layer
#         self.dropout = nn.Dropout(0.1)
        
#     def forward(self, x):
#         batch_size, seq_len, embed_dim = x.size()
        
#         # Linear transformation and split into multiple heads
#         query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # B x num_heads x seq_len x head_dim
#         key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # B x num_heads x seq_len x head_dim
#         value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # B x num_heads x seq_len x head_dim
        
#         # Compute scaled dot-product attention scores
#         scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5) # B x num_heads x seq_len x seq_len
        
#         # Apply softmax to obtain attention weights
#         attention_weights = F.softmax(scores, dim=-1) # B x num_heads x seq_len x seq_len
        
#         # Apply dropout
#         attention_weights = self.dropout(attention_weights)
        
#         # Weighted sum of values with attention weights
#         attention_output = torch.matmul(attention_weights, value) # B x num_heads x seq_len x head_dim
        
#         # Concatenate heads and reshape
#         attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim) # B x seq_len x embed_dim
        
#         return attention_output

# class MyModel(nn.Module):
#     def __init__(self, d_model=768, d_ffn=256, seq_len=2001, gmlp_layers=1, batch_first=True, flag_pool='sap', use_self_attention=True, num_attention_heads=8):
#         super(MyModel, self).__init__()

#         self.batch_first = batch_first
#         self.flag_pool = flag_pool
#         self.d_model = d_model
#         self.use_self_attention = use_self_attention
#         self.num_attention_heads = num_attention_heads

#         self.SelfWeightedPooling = SelfWeightedPooling(self.d_model, mean_only=True)

#         if d_ffn > 0:
#             pass
#         elif d_ffn < 0:
#             # If emb_dim < 0, we will reduce dim by emb_dim. Like -2 will be dim/2
#             d_ffn = int(d_model / abs(d_ffn))

#         layers = []
#         for i in range(gmlp_layers):
#             layers.append(GMLPBlock(d_model, d_ffn, seq_len))
#         self.layers = nn.Sequential(*layers)

#         self.fc1 = nn.Linear(d_model, d_ffn, bias=False)

#         # Additional linear layers for classification
#         self.fc2 = nn.Linear(d_ffn, 33)

#         # Self-attention layer
#         if self.use_self_attention:
#             self.self_attention = SelfAttention(d_model, num_attention_heads)

#     def forward(self, x):
#         x = self.layers(x)

#         if self.use_self_attention:
#             x = self.self_attention(x)

#         # Pool for utt
#         if self.flag_pool == "ap":  # Average pooling
#             x = x.mean(dim=1)
#         elif self.flag_pool == "sap":
#             x = self.SelfWeightedPooling(x)
#         else:
#             pass

#         x = self.fc1(x)
#         x = F.gelu(x)
#         x = self.fc2(x)
#         x = torch.sigmoid(x)

#         # print(f'In SimpleNN model with self-attention, output of type {type(x)} with size {x.size()}')

#         return x










# ===================================  conformer model   =============================
import torch
import math

# class RelativePositionEncoding(nn.Module):
#     def __init__(self, num_heads, hidden_dim, max_len=5000):
#         super(RelativePositionEncoding, self).__init__()
#         self.num_heads = num_heads
#         self.hidden_dim = hidden_dim
        
#         # We use the same approach as the Transformer to create sinusoidal embeddings
#         self.pos_embedding = self._generate_relative_positions_encoding(max_len, hidden_dim)
    
#     def _generate_relative_positions_encoding(self, max_len, hidden_dim):
#         """
#         Generate sinusoidal relative position encodings.
#         """
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * -(math.log(10000.0) / hidden_dim))
#         encoding = torch.zeros(max_len, hidden_dim)
#         encoding[:, 0::2] = torch.sin(position * div_term)
#         encoding[:, 1::2] = torch.cos(position * div_term)
        
#         # Add an extra dimension for each head
#         encoding = encoding.unsqueeze(0).repeat(self.num_heads, 1, 1)
#         return encoding

#     def forward(self, length):
#         """
#         Returns the relative position encoding up to a given length.
#         """
#         return self.pos_embedding[:, :length, :]


# class RelativeMultiheadAttention(nn.Module):
#     def __init__(self, input_dim, num_heads, max_len=5000):
#         super(RelativeMultiheadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.hidden_dim = input_dim
#         self.head_dim = input_dim // num_heads

#         # Linear transformations for queries, keys, and values
#         self.query_linear = nn.Linear(input_dim, input_dim)
#         self.key_linear = nn.Linear(input_dim, input_dim)
#         self.value_linear = nn.Linear(input_dim, input_dim)
        
#         # Output linear layer
#         self.out_linear = nn.Linear(input_dim, input_dim)
        
#         # Relative position encoding
#         self.relative_pos_encoding = RelativePositionEncoding(num_heads, self.head_dim, max_len)

#     def forward(self, query, key, value, mask=None):
#         batch_size, seq_len, _ = query.size()

#         # Linear projections
#         Q = self.query_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         K = self.key_linear(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         V = self.value_linear(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

#         # Relative positional encoding
#         relative_pos = self.relative_pos_encoding(seq_len).to(Q.device)
#         # relative_pos = relative_pos.to(Q.device)

#         # Compute attention scores
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # Standard attention
#         # Incorporating relative position encoding
#         attn_scores = attn_scores + torch.matmul(Q, relative_pos.transpose(-2, -1))

#         if mask is not None:
#             attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
#         # Softmax attention weights
#         attn_weights = torch.softmax(attn_scores, dim=-1)
        
#         # Attention output
#         output = torch.matmul(attn_weights, V)
        
#         # Concatenate heads and pass through output linear layer
#         output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
#         output = self.out_linear(output)
#         return output


# class ConformerBlock(nn.Module):
#     def __init__(self, input_dim, num_heads, expansion_factor=4, max_len=5000):
#         super(ConformerBlock, self).__init__()
#         self.attn = RelativeMultiheadAttention(input_dim, num_heads, max_len)
#         self.ffn = nn.Sequential(
#             nn.Linear(input_dim, input_dim * expansion_factor),
#             nn.ReLU(),
#             nn.Linear(input_dim * expansion_factor, input_dim)
#         )
#         self.norm1 = nn.LayerNorm(input_dim)
#         self.norm2 = nn.LayerNorm(input_dim)
#         self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
    
#     def forward(self, x):
#         # Multi-head self-attention with relative position encoding
#         attn_out = self.attn(x, x, x)
#         x = self.norm1(x + attn_out)  # Add & Norm

#         # Feed-forward
#         ff_out = self.ffn(x)
#         x = self.norm2(x + ff_out)  # Add & Norm

#         # Convolutional module
#         conv_out = self.conv(x.permute(0, 2, 1))  # Conv1d expects (batch, channels, seq_len)
#         return conv_out.permute(0, 2, 1)  # Permute back to (batch, seq_len, channels)


# class ConformerBlock(nn.Module):
#     def __init__(self, input_dim, num_heads, expansion_factor=4):
#         super(ConformerBlock, self).__init__()
#         self.attn = nn.MultiheadAttention(input_dim, num_heads)
#         self.ffn = nn.Sequential(
#             nn.Linear(input_dim, input_dim * expansion_factor),
#             nn.ReLU(),
#             nn.Linear(input_dim * expansion_factor, input_dim)
#         )
#         self.norm1 = nn.LayerNorm(input_dim)
#         self.norm2 = nn.LayerNorm(input_dim)
#         self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
    
#     def forward(self, x):
#         # Multi-head self-attention
#         attn_out, _ = self.attn(x, x, x)
#         x = self.norm1(x + attn_out)  # Add & Norm

#         # Feed-forward
#         ff_out = self.ffn(x)
#         x = self.norm2(x + ff_out)  # Add & Norm

#         # Convolutional module
#         conv_out = self.conv(x.permute(0, 2, 1))  # Conv1d expects (batch, channels, seq_len)
#         return conv_out.permute(0, 2, 1)  # Permute back to (batch, seq_len, channels)




# class BiLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers=1):
#         super(BiLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)

#     def forward(self, x):
#         return self.lstm(x)[0]


# class FullyConnectedBlock(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(FullyConnectedBlock, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.fc(x)
#         return self.sigmoid(x)



# class PoolingIntegration(nn.Module):
#     def __init__(self, omega1=0.4, omega2=0.4, omega3=0.2):
#         super(PoolingIntegration, self).__init__()
#         self.omega1 = omega1
#         self.omega2 = omega2
#         self.omega3 = omega3
    
#     def forward(self, x):
#         power_pooling = torch.pow(x, 2).sum(dim=1)
#         auto_softmax_pooling = torch.softmax(x, dim=1).sum(dim=1)
#         max_pooling = torch.max(x, dim=1)[0]

#         utterance_score = (self.omega1 * power_pooling + 
#                            self.omega2 * auto_softmax_pooling + 
#                            self.omega3 * max_pooling)
#         return utterance_score




# class SpoofingDetectionModel(nn.Module):
#     def __init__(self, feature_dim, num_heads, hidden_dim, num_classes):
#         super(SpoofingDetectionModel, self).__init__()
#         # self.selcnn = SELCNN(input_dim, 64)
#         self.SelfWeightedPooling = SelfWeightedPooling(feature_dim, mean_only=True)
#         self.projector = nn.Linear(feature_dim, 64)  # Project 768 -> conformer_dim
        
#         self.conformer = ConformerBlock(64, num_heads)
#         self.bilstm = BiLSTM(64, hidden_dim)
#         self.fc_block = FullyConnectedBlock(hidden_dim * 2, num_classes)  # LSTM is bidirectional
#         # self.pooling_integration = PoolingIntegration()

#     def forward(self, x):
#         # Feature Extraction and Refinement
#         # x = self.selcnn(x)
#         # print(f"SpoofingDetectionModel , x size = { x.size()}")
#         x = self.SelfWeightedPooling(x)
#         x = self.projector(x)
#         x = self.conformer(x.unsqueeze(1))
        
#         # Contextualization
#         x = self.bilstm(x)
        
#         # Classification
#         segment_score = self.fc_block(x).squeeze(1)
        
#         # Pooling and Final Score Calculation
#         # utterance_score = self.pooling_integration(segment_score)
#         # utterance_score = 0
        
#         # return segment_score, utterance_score
#         return segment_score







# import torch
# import torch.nn as nn
# import torchaudio.models as tam
# class MyUpdatedSpoofingDetectionModel(nn.Module):
#     def __init__(self, feature_dim, num_heads, hidden_dim, num_classes, depthwise_conv_kernel_size=31):
#         super(MyUpdatedSpoofingDetectionModel, self).__init__()
        
#         # Define the Conformer model from torchaudio
#         self.conformer = tam.Conformer(
#             input_dim=feature_dim,
#             num_heads=num_heads,
#             ffn_dim=hidden_dim,  # Feed-forward network dimension (for consistency)
#             num_layers=1,  # Example, adjust as needed
#             depthwise_conv_kernel_size=depthwise_conv_kernel_size,  # Set the kernel size for depthwise convolution
#             dropout=0.5,
#             use_group_norm= False, 
#             convolution_first= False
#         )
        
#         # Global pooling layer (SelfWeightedPooling)
#         self.pooling = SelfWeightedPooling(feature_dim, mean_only=True)  # Pool across sequence dimension
        
#         # Add a feedforward block for feature refinement before classification
#         self.fc_refinement = nn.Sequential(
#             nn.Linear(feature_dim, hidden_dim),  # Refined hidden dimension for classification
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim),
#             nn.Dropout(0.3),  # Dropout for regularization

#             nn.Linear(hidden_dim, num_classes),  # Final output layer
#             nn.Sigmoid(),
#         )
        
#     def forward(self, x, lengths):
#         # print(f" x size before conformer = {x.size()}")
        
#         # Apply Conformer model
#         x, _ = self.conformer(x, lengths)  # The second returned value is the sequence lengths
#         # print(f" x size after conformer = {x.size()}")
        
#         # Apply global pooling across the sequence dimension (SelfWeightedPooling)
#         x = self.pooling(x)  # Now x is (batch_size, hidden_dim, 1)
#         # print(f" x size after pooling = {x.size()}")
        
#         # Refine features before classification using the fc_refinement block
#         segment_score = self.fc_refinement(x)
#         # print(f" x size after fc_refinement = {segment_score.size()}")
        
#         # Return the classification output
#         return segment_score














# ===============================================================================================================



# =================================== conformer with relative positional embedding ====

class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        # if torch.cuda.is_available():
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()
        # else:
        #     final_mat = torch.LongTensor(final_mat)
        #     embeddings = self.embeddings_table[final_mat]

        return embeddings

class RelativeMultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, dropout_prob,mask = None):
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2)) 

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size*self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)
        self.dropout.p=dropout_prob
        attn = self.dropout(torch.softmax(attn, dim = -1))

        #attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size*self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x


from typing import Optional, Tuple

import torch


__all__ = ["Conformer"]


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


class _ConvolutionModule(torch.nn.Module):
    r"""Conformer convolution module.

    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
        use_group_norm (bool, optional): use GroupNorm rather than BatchNorm. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            torch.nn.GroupNorm(num_groups=1, num_channels=num_channels)
            if use_group_norm
            else torch.nn.BatchNorm1d(num_channels),
            torch.nn.SiLU(),
            torch.nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor,dropout_prob) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        self.sequential[-1].p=dropout_prob
        x = self.sequential(x)
        return x.transpose(1, 2)


class _FeedForwardModule(torch.nn.Module):
    r"""Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, input_dim, bias=True),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor,dropout_prob) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        self.sequential[3].p=dropout_prob
        self.sequential[5].p=dropout_prob
        return self.sequential(input)


class ConformerLayer(torch.nn.Module):
    r"""Conformer layer that constitutes Conformer.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
        device: torch.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> None:
        super().__init__()

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        # self.self_attn = torch.nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        # self.self_attn = RelPartialLearnableMultiHeadAttn( num_attention_heads,input_dim,64, dropout=dropout)
        self.self_attn = RelativeMultiHeadAttentionLayer(input_dim, num_attention_heads, dropout, device)

        self.self_attn_dropout = torch.nn.Dropout(dropout)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_convolution(self, input: torch.Tensor,dropout_prob) -> torch.Tensor:
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input,dropout_prob)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor],dropout_prob) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.

        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        # self_attn_dropout = torch.nn.Dropout(dropout_prob)

        residual = input
        x = self.ffn1(input,dropout_prob)
        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_convolution(x,dropout_prob)

        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            dropout_prob=dropout_prob,
        ) # code from https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py
        # x, _ = self.self_attn(x) # code from transformer xl
        self.self_attn_dropout.p=dropout_prob
        x = self.self_attn_dropout(x)
        # x = self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_convolution(x,dropout_prob)

        residual = x
        x = self.ffn2(x,dropout_prob)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x


class Conformer(torch.nn.Module):
    r"""Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    :cite:`gulati2020conformer`.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)

    Examples:
        >>> conformer = Conformer(
        >>>     input_dim=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ):
        super().__init__()

        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: torch.Tensor, lengths: torch.Tensor,dropout_prob) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """
        encoder_padding_mask = _lengths_to_padding_mask(lengths)

        x = input.transpose(0, 1)
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask,dropout_prob)
        return x.transpose(0, 1), lengths


class MyUpdatedSpoofingDetectionModel(nn.Module):
    def __init__(self, feature_dim, num_heads, hidden_dim, num_classes,max_dropout=0.5, depthwise_conv_kernel_size=31,conformer_layers=1):
        super(MyUpdatedSpoofingDetectionModel, self).__init__()
        
        self.max_dropout=max_dropout
        # Define the Conformer model from torchaudio
        self.conformer = Conformer(
            input_dim=feature_dim,
            num_heads=num_heads,
            ffn_dim=hidden_dim,  # Feed-forward network dimension (for consistency)
            num_layers=conformer_layers,  # Example, adjust as needed
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,  # Set the kernel size for depthwise convolution
            dropout=0.5,
            use_group_norm= False, 
            convolution_first= False
        )
        
        # Global pooling layer (SelfWeightedPooling)
        self.pooling = SelfWeightedPooling(feature_dim, mean_only=True)  # Pool across sequence dimension
        
        # Add a feedforward block for feature refinement before classification
        self.fc_refinement = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),  # Refined hidden dimension for classification
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),  # Dropout for regularization

            nn.Linear(hidden_dim, num_classes),  # Final output layer
            nn.Sigmoid(),
        )
        
    def forward(self, x, lengths,dropout_prob):
        # print(f" x size before conformer = {x.size()}")
        
        # Apply Conformer model
        x, _ = self.conformer(x, lengths,dropout_prob)  # The second returned value is the sequence lengths
        # print(f" x size after conformer = {x.size()}")
        
        # Apply global pooling across the sequence dimension (SelfWeightedPooling)
        x = self.pooling(x)  # Now x is (batch_size, hidden_dim, 1)
        # print(f" x size after pooling = {x.size()}")
        
        # Update the dropout probability dynamically
        self.fc_refinement[3].p = dropout_prob  # Update the dropout layer's probability

        # Refine features before classification using the fc_refinement block
        segment_score = self.fc_refinement(x)
        # print(f" x size after fc_refinement = {segment_score.size()}")
        
        # Return the classification output
        return segment_score

    def adjust_dropout(self, epoch, total_epochs):
        # Cosine annealing for dropout probability
        return self.max_dropout * (1 + math.cos(math.pi * epoch / total_epochs)) / 2






# ================================================================================================
# ================================================================================================
# ================================================================================================
# ================================================================================================



# import torch
# import torch.nn as nn
# import math

# class RelativePosition(nn.Module):
#     def __init__(self, num_units, max_relative_position):
#         super().__init__()
#         self.num_units = num_units
#         self.max_relative_position = max_relative_position
#         self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
#         nn.init.xavier_uniform_(self.embeddings_table)

#     def forward(self, length_q, length_k):
#         range_vec_q = torch.arange(length_q)
#         range_vec_k = torch.arange(length_k)
#         distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
#         distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
#         final_mat = distance_mat_clipped + self.max_relative_position
#         if torch.cuda.is_available():
#             final_mat = torch.LongTensor(final_mat).cuda()
#             embeddings = self.embeddings_table[final_mat].cuda()
#         else:
#             final_mat = torch.LongTensor(final_mat)
#             embeddings = self.embeddings_table[final_mat]

#         return embeddings

# class RelativeMultiHeadAttentionLayer(nn.Module):
#     def __init__(self, hid_dim, n_heads, max_dropout, device):
#         super().__init__()

#         assert hid_dim % n_heads == 0

#         self.hid_dim = hid_dim
#         self.n_heads = n_heads
#         self.head_dim = hid_dim // n_heads
#         self.max_relative_position = 2

#         self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
#         self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

#         self.fc_q = nn.Linear(hid_dim, hid_dim)
#         self.fc_k = nn.Linear(hid_dim, hid_dim)
#         self.fc_v = nn.Linear(hid_dim, hid_dim)

#         self.fc_o = nn.Linear(hid_dim, hid_dim)

#         self.max_dropout = max_dropout
#         self.dropout = nn.Dropout(0.0)  # Initial dropout 0.0

#         self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

#     def forward(self, query, key, value, mask=None, dropout_prob=0.0):
#         self.dropout.p = dropout_prob  # Dynamically set dropout probability
#         batch_size = query.shape[0]
#         len_k = key.shape[1]
#         len_q = query.shape[1]
#         len_v = value.shape[1]

#         query = self.fc_q(query)
#         key = self.fc_k(key)
#         value = self.fc_v(value)

#         r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#         r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#         attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

#         r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size * self.n_heads, self.head_dim)
#         r_k2 = self.relative_position_k(len_q, len_k)
#         attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
#         attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
#         attn = (attn1 + attn2) / self.scale

#         if mask is not None:
#             attn = attn.masked_fill(mask == 0, -1e10)

#         attn = self.dropout(torch.softmax(attn, dim=-1))  # Apply dropout here

#         # attn = [batch size, n heads, query len, key len]
#         r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#         weight1 = torch.matmul(attn, r_v1)
#         r_v2 = self.relative_position_v(len_q, len_v)
#         weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size * self.n_heads, len_k)
#         weight2 = torch.matmul(weight2, r_v2)
#         weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

#         x = weight1 + weight2
#         x = x.permute(0, 2, 1, 3).contiguous()
#         x = x.view(batch_size, -1, self.hid_dim)
#         x = self.fc_o(x)

#         return x


# class ConformerLayer(nn.Module):
#     def __init__(self, input_dim, ffn_dim, num_attention_heads, max_dropout, dropout=0.0, device=torch.device('cuda')):
#         super().__init__()

#         self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
#         self.self_attn_layer_norm = nn.LayerNorm(input_dim)
#         self.self_attn = RelativeMultiHeadAttentionLayer(input_dim, num_attention_heads, max_dropout, device)
#         self.self_attn_dropout = nn.Dropout(dropout)
#         self.conv_module = _ConvolutionModule(input_dim=input_dim, num_channels=input_dim, depthwise_kernel_size=31, dropout=dropout)
#         self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
#         self.final_layer_norm = nn.LayerNorm(input_dim)

#     def forward(self, input, lengths, dropout_prob=0.0):
#         residual = input
#         x = self.ffn1(input)
#         x = x * 0.5 + residual

#         residual = x
#         x = self.self_attn_layer_norm(x)
#         x = self.self_attn(query=x, key=x, value=x, dropout_prob=dropout_prob)
#         x = self.self_attn_dropout(x)
#         x = x + residual

#         x = self.conv_module(x)
#         residual = x
#         x = self.ffn2(x)
#         x = x * 0.5 + residual
#         x = self.final_layer_norm(x)
#         return x


# class Conformer(nn.Module):
#     def __init__(self, input_dim, num_heads, ffn_dim, num_layers, max_dropout, dropout=0.0):
#         super().__init__()

#         self.conformer_layers = nn.ModuleList([
#             ConformerLayer(input_dim, ffn_dim, num_heads, max_dropout, dropout)
#             for _ in range(num_layers)
#         ])

#     def adjust_dropout(self, epoch, total_epochs, max_dropout):
#         # Cosine annealing for dropout probability
#         return max_dropout * (1 + math.cos(math.pi * epoch / total_epochs)) / 2

#     def forward(self, input, lengths, epoch, total_epochs):
#         dropout_prob = self.adjust_dropout(epoch, total_epochs, max_dropout=0.5)  # Adjust dropout for the epoch
#         x = input.transpose(0, 1)  # Transpose for (T, B, D)
#         for layer in self.conformer_layers:
#             x = layer(x, lengths, dropout_prob=dropout_prob)
#         return x.transpose(0, 1), lengths


# class MyUpdatedSpoofingDetectionModel(nn.Module):
#     def __init__(self, feature_dim, num_heads, hidden_dim, num_classes, depthwise_conv_kernel_size=31, conformer_layers=1, max_dropout=0.5):
#         super(MyUpdatedSpoofingDetectionModel, self).__init__()

#         # Define the Conformer model with dropout scheduling
#         self.conformer = Conformer(
#             input_dim=feature_dim,
#             num_heads=num_heads,
#             ffn_dim=hidden_dim,
#             num_layers=conformer_layers,
#             max_dropout=max_dropout,
#             dropout=0.5  # Initial dropout
#         )

#         # Global pooling layer (SelfWeightedPooling)
#         self.pooling = SelfWeightedPooling(feature_dim, mean_only=True)

#         # Add a feedforward block for feature refinement before classification
#         self.fc_refinement = nn.Sequential(
#             nn.Linear(feature_dim, hidden_dim),
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim),
#             nn.Dropout(0.3),
#             nn.Linear(hidden_dim, num_classes),
#             nn.Sigmoid(),
#         )

#     def forward(self, x, lengths, epoch, total_epochs):
#         print(f" x size before conformer = {x.size()}")

#         # Apply Conformer model with dynamic dropout scheduling
#         x, _ = self.conformer(x, lengths, epoch, total_epochs)

#         print(f" x size after conformer = {x.size()}")

#         # Apply pooling
#         pooled_features = self.pooling(x)
#         print(f"pooled_features size = {pooled_features.size()}")

#         # Feature refinement and classification
#         out = self.fc_refinement(pooled_features.squeeze(2))  # Remove singleton dimension (1)
#         return out

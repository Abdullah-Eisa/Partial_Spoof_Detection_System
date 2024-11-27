
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



import torch
import torch.nn as nn
import torchaudio.models as tam
class MyUpdatedSpoofingDetectionModel(nn.Module):
    def __init__(self, feature_dim, num_heads, hidden_dim, num_classes, depthwise_conv_kernel_size=31):
        super(MyUpdatedSpoofingDetectionModel, self).__init__()
        
        # Define the Conformer model from torchaudio
        self.conformer = tam.Conformer(
            input_dim=feature_dim,
            num_heads=num_heads,
            ffn_dim=hidden_dim,  # Feed-forward network dimension (for consistency)
            num_layers=1,  # Example, adjust as needed
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
        
    def forward(self, x, lengths):
        # print(f" x size before conformer = {x.size()}")
        
        # Apply Conformer model
        x, _ = self.conformer(x, lengths)  # The second returned value is the sequence lengths
        # print(f" x size after conformer = {x.size()}")
        
        # Apply global pooling across the sequence dimension (SelfWeightedPooling)
        x = self.pooling(x)  # Now x is (batch_size, hidden_dim, 1)
        # print(f" x size after pooling = {x.size()}")
        
        # Refine features before classification using the fc_refinement block
        segment_score = self.fc_refinement(x)
        # print(f" x size after fc_refinement = {segment_score.size()}")
        
        # Return the classification output
        return segment_score

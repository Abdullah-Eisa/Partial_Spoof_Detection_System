
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



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from gmlp import GMLPBlock  # Ensure that you have this import if needed
from utils import *  # Ensure that you have this import if needed

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Define linear transformations for query, key, and value
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Linear transformation and split into multiple heads
        query = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # B x num_heads x seq_len x head_dim
        key = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # B x num_heads x seq_len x head_dim
        value = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # B x num_heads x seq_len x head_dim
        
        # Compute scaled dot-product attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5) # B x num_heads x seq_len x seq_len
        
        # Apply softmax to obtain attention weights
        attention_weights = F.softmax(scores, dim=-1) # B x num_heads x seq_len x seq_len
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum of values with attention weights
        attention_output = torch.matmul(attention_weights, value) # B x num_heads x seq_len x head_dim
        
        # Concatenate heads and reshape
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim) # B x seq_len x embed_dim
        
        return attention_output

class MyModel(nn.Module):
    def __init__(self, d_model=768, d_ffn=256, seq_len=2001, gmlp_layers=1, batch_first=True, flag_pool='sap', use_self_attention=True, num_attention_heads=8):
        super(MyModel, self).__init__()

        self.batch_first = batch_first
        self.flag_pool = flag_pool
        self.d_model = d_model
        self.use_self_attention = use_self_attention
        self.num_attention_heads = num_attention_heads

        self.SelfWeightedPooling = SelfWeightedPooling(self.d_model, mean_only=True)

        if d_ffn > 0:
            pass
        elif d_ffn < 0:
            # If emb_dim < 0, we will reduce dim by emb_dim. Like -2 will be dim/2
            d_ffn = int(d_model / abs(d_ffn))

        layers = []
        for i in range(gmlp_layers):
            layers.append(GMLPBlock(d_model, d_ffn, seq_len))
        self.layers = nn.Sequential(*layers)

        self.fc1 = nn.Linear(d_model, d_ffn, bias=False)

        # Additional linear layers for classification
        self.fc2 = nn.Linear(d_ffn, 33)

        # Self-attention layer
        if self.use_self_attention:
            self.self_attention = SelfAttention(d_model, num_attention_heads)

    def forward(self, x):
        x = self.layers(x)

        if self.use_self_attention:
            x = self.self_attention(x)

        # Pool for utt
        if self.flag_pool == "ap":  # Average pooling
            x = x.mean(dim=1)
        elif self.flag_pool == "sap":
            x = self.SelfWeightedPooling(x)
        else:
            pass

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        # print(f'In SimpleNN model with self-attention, output of type {type(x)} with size {x.size()}')

        return x







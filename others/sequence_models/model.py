import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from utils.utils import *
import torch.nn.init as torch_init
import os
import torch.optim as optim
import torchaudio.models as tam

# Import sequence models
from sequence_models import (
    LSTMSequenceModel,
    TransformerSequenceModel,
    CNNSequenceModel,
    TCNSequenceModel,
    create_sequence_model
)

# ============================================================================================
# SAP = SelfWeightedPooling
# code adapted from: https://github.com/nii-yamagishilab/PartialSpoof/blob/847347aaec6f65c3c6d2f17c63515b826b94feb3/project-NN-Pytorch-scripts.202102/sandbox/block_nn.py#L709

class SelfWeightedPooling(nn.Module):
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
        self.mm_weights = nn.Parameter(
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
        attentions = F.softmax(torch.tanh(weights),dim=1)        
        
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
# Updated Binary Classification Model with Multiple Sequence Model Options

class BinarySpoofingClassificationModel(nn.Module):
    def __init__(self, feature_dim, num_heads, hidden_dim, max_dropout=0.2, 
                 depthwise_conv_kernel_size=31, conformer_layers=1, max_pooling_factor=3,
                 sequence_model_type='conformer', sequence_model_config=None):
        """
        Binary Spoofing Classification Model with flexible sequence modeling
        
        Args:
            feature_dim (int): Feature dimension from feature extractor
            num_heads (int): Number of attention heads
            hidden_dim (int): Hidden dimension for feed-forward layers
            max_dropout (float): Maximum dropout probability
            depthwise_conv_kernel_size (int): Kernel size for depthwise convolution (conformer only)
            conformer_layers (int): Number of conformer/sequence layers
            max_pooling_factor (int): Max pooling factor (None to disable)
            sequence_model_type (str): Type of sequence model 
                ('conformer', 'lstm', 'transformer', 'cnn', 'tcn')
            sequence_model_config (dict): Additional config for sequence model
        """
        super(BinarySpoofingClassificationModel, self).__init__()

        self.max_pooling_factor = max_pooling_factor
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.max_dropout = max_dropout
        self.sequence_model_type = sequence_model_type.lower()

        # Apply max pooling if specified
        if self.max_pooling_factor is not None:
            self.max_pooling = nn.MaxPool1d(
                kernel_size=self.max_pooling_factor, 
                stride=self.max_pooling_factor
            )
            self.feature_dim = feature_dim // self.max_pooling_factor
        else:
            self.max_pooling = None
        
        print(f"Sequence Model Type: {self.sequence_model_type}")
        print(f"Feature dim after pooling: {self.feature_dim}")
        print(f"Max pooling: {self.max_pooling}")
        
        # Initialize sequence model based on type
        if self.sequence_model_type == 'conformer':
            # Original Conformer model from torchaudio
            self.sequence_model = tam.Conformer(
                input_dim=self.feature_dim,
                num_heads=self.num_heads,
                ffn_dim=hidden_dim,
                num_layers=conformer_layers,
                depthwise_conv_kernel_size=depthwise_conv_kernel_size,
                dropout=0.2,
                use_group_norm=False, 
                convolution_first=False
            )
            self.sequence_output_dim = self.feature_dim
            
        elif self.sequence_model_type == 'lstm':
            # LSTM sequence model
            seq_config = sequence_model_config or {}
            seq_config.setdefault('hidden_dim', hidden_dim)
            seq_config.setdefault('num_layers', conformer_layers)
            seq_config.setdefault('dropout', 0.2)
            
            self.sequence_model = LSTMSequenceModel(
                input_dim=self.feature_dim,
                **seq_config
            )
            self.sequence_output_dim = self.sequence_model.output_dim
            
        elif self.sequence_model_type == 'transformer':
            # Transformer sequence model
            seq_config = sequence_model_config or {}
            seq_config.setdefault('num_heads', num_heads)
            seq_config.setdefault('hidden_dim', hidden_dim)
            seq_config.setdefault('num_layers', conformer_layers)
            seq_config.setdefault('dropout', 0.2)
            
            self.sequence_model = TransformerSequenceModel(
                input_dim=self.feature_dim,
                **seq_config
            )
            self.sequence_output_dim = self.sequence_model.output_dim
            
        elif self.sequence_model_type == 'cnn':
            # CNN sequence model
            seq_config = sequence_model_config or {}
            seq_config.setdefault('hidden_dim', hidden_dim)
            seq_config.setdefault('num_layers', conformer_layers)
            seq_config.setdefault('kernel_size', 3)
            seq_config.setdefault('dropout', 0.2)
            
            self.sequence_model = CNNSequenceModel(
                input_dim=self.feature_dim,
                **seq_config
            )
            self.sequence_output_dim = self.sequence_model.output_dim
            
        elif self.sequence_model_type == 'tcn':
            # TCN sequence model
            seq_config = sequence_model_config or {}
            seq_config.setdefault('hidden_dim', hidden_dim)
            seq_config.setdefault('num_layers', conformer_layers)
            seq_config.setdefault('kernel_size', 3)
            seq_config.setdefault('dropout', 0.2)
            
            self.sequence_model = TCNSequenceModel(
                input_dim=self.feature_dim,
                **seq_config
            )
            self.sequence_output_dim = self.sequence_model.output_dim
            
        else:
            raise ValueError(
                f"Unknown sequence_model_type: {self.sequence_model_type}. "
                f"Supported types: 'conformer', 'lstm', 'transformer', 'cnn', 'tcn'"
            )
        
        print(f"Sequence model output dim: {self.sequence_output_dim}")
        
        # Global pooling layer (SelfWeightedPooling)
        self.pooling = SelfWeightedPooling(self.sequence_output_dim, mean_only=True)
        
        # Feed-forward classification head
        self.fc_refinement = nn.Sequential(
            nn.Linear(self.sequence_output_dim, hidden_dim),
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

    def initialize_weights(self, m, bias_value=0.005):
        """Custom initialization for He and Xavier"""
        if isinstance(m, nn.Linear):
            if hasattr(m, 'activation') and isinstance(m.activation, nn.ReLU):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif hasattr(m, 'activation') and isinstance(m.activation, nn.GELU):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif hasattr(m, 'activation') and isinstance(m.activation, (nn.Tanh, nn.Sigmoid)):
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_value)

        elif isinstance(m, nn.Conv1d):
            if hasattr(m, 'activation') and isinstance(m.activation, nn.ReLU):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif hasattr(m, 'activation') and isinstance(m.activation, (nn.Tanh, nn.Sigmoid)):
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, bias_value)

    def forward(self, x, lengths, dropout_prob):
        """
        Forward pass
        
        Args:
            x: Input features (batch_size, seq_len, feature_dim)
            lengths: Sequence lengths
            dropout_prob: Dropout probability for classification head
            
        Returns:
            utt_score: Binary classification score
        """
        # Apply max pooling if enabled
        if self.max_pooling is not None:
            x = self.max_pooling(x)

        # Apply sequence model
        if self.sequence_model_type == 'conformer':
            x, _ = self.sequence_model(x, lengths)
        else:
            x, _ = self.sequence_model(x, lengths)
        
        # Apply global pooling across the sequence dimension
        x = self.pooling(x)

        # Update dropout probability dynamically in classification head
        self.fc_refinement[3].p = dropout_prob
        self.fc_refinement[7].p = dropout_prob
        self.fc_refinement[11].p = dropout_prob

        # Classification
        utt_score = self.fc_refinement(x)
        return utt_score
        
    def adjust_dropout(self, epoch, total_epochs):
        """Cosine annealing for dropout probability"""
        return self.max_dropout * (1 + math.cos(math.pi * epoch / total_epochs)) / 2


# ===========================================================================================================================
# ===========================================================================================================================
# Model Initialization Function

def initialize_models(ssl_ckpt_path, save_feature_extractor=False,
                      feature_dim=768, num_heads=8, hidden_dim=128, max_dropout=0.2, 
                      depthwise_conv_kernel_size=31, conformer_layers=1, max_pooling_factor=3,
                      sequence_model_type='conformer', sequence_model_config=None,
                      LEARNING_RATE=0.0001, DEVICE='cpu'):
    """
    Initialize the model, feature extractor, and optimizer
    
    Args:
        ssl_ckpt_path (str): Path to SSL checkpoint
        save_feature_extractor (bool): Whether to save/train feature extractor
        feature_dim (int): Feature dimension
        num_heads (int): Number of attention heads
        hidden_dim (int): Hidden dimension
        max_dropout (float): Maximum dropout
        depthwise_conv_kernel_size (int): Kernel size for depthwise conv (conformer only)
        conformer_layers (int): Number of sequence model layers
        max_pooling_factor (int): Max pooling factor
        sequence_model_type (str): Type of sequence model
        sequence_model_config (dict): Additional sequence model config
        LEARNING_RATE (float): Learning rate
        DEVICE (str): Device to use
        
    Returns:
        PS_Model: Classification model
        feature_extractor: Feature extraction model
        optimizer: Optimizer
    """
    # Initialize feature extractor
    if os.path.exists(ssl_ckpt_path):
        feature_extractor = torch.hub.load('s3prl/s3prl', 'wav2vec2', 
                                          model_path=ssl_ckpt_path).to(DEVICE)
    else:
        ssl_ckpt_path = os.path.join(os.getcwd(), 'models/w2v_large_lv_fsh_swbd_cv.pt')
        feature_extractor = torch.hub.load('s3prl/s3prl', 'wav2vec2', 
                                          model_path=ssl_ckpt_path).to(DEVICE)

    # Initialize Binary Spoofing Classification Model
    PS_Model = BinarySpoofingClassificationModel(
        feature_dim=feature_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        max_dropout=max_dropout,
        depthwise_conv_kernel_size=depthwise_conv_kernel_size,
        conformer_layers=conformer_layers,
        max_pooling_factor=max_pooling_factor,
        sequence_model_type=sequence_model_type,
        sequence_model_config=sequence_model_config
    ).to(DEVICE)

    # Freeze feature extractor if necessary
    if save_feature_extractor:
        for name, param in feature_extractor.named_parameters():
            if 'final_proj' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    # Optimizer setup
    if save_feature_extractor:
        optimizer = optim.AdamW(
            [{'params': feature_extractor.parameters(), 'lr': 0.00005},
             {'params': PS_Model.parameters()}], 
            lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = optim.AdamW(
            PS_Model.parameters(), 
            lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)

    return PS_Model, feature_extractor, optimizer


# ===========================================================================================================================
# Utility Functions

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
    patience = 5
    threshold = 0.009
    min_lr = 0.00001

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience, 
        threshold=threshold, min_lr=min_lr)
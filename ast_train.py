import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from utils import *
import random
import torchaudio.transforms as T
from torch.utils.data import Dataset
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader , ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp


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



from utils import load_json_dictionary
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


# -*- coding: utf-8 -*-
# @Time    : 6/10/21 5:04 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ast_models.py

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
# os.environ['TORCH_HOME'] = '../../pretrained_models'
import timm
from timm.models.layers import to_2tuple,trunc_normal_

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=1, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=True):

        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists('../../pretrained_models/audioset_10_10_0.4593.pth') == False:
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='../../pretrained_models/audioset_10_10_0.4593.pth')
            sd = torch.load('../../pretrained_models/audioset_10_10_0.4593.pth', map_location=device)
            audio_model = ASTModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            # otherwise interpolate
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: prediction
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        return x



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
        [{'params': feature_extractor.parameters(), 'lr': 0.0001},
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
    # LR_SCHEDULER = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # def lr_lambda(epoch):
    #     if epoch < 20:
    #         return 1 - (epoch / 20)  # Decrease linearly from 1 to 0 over 30 epochs
    #     else:
    #         return 1 + (epoch - 20) / 10  # Increase linearly after epoch 30

    # LR_SCHEDULER = lr_scheduler.LambdaLR(optimizer, lr_lambda)

    factor = 2/3
    patience = 5   # Number of epochs with no improvement after which learning rate will be reduced
    threshold=0.005
    min_lr = 0.00001

    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, threshold=threshold, min_lr=min_lr)



# if __name__ == '__main__':

#     import torch
#     use_cuda= True
#     use_cuda =  use_cuda and torch.cuda.is_available()
#     DEVICE = torch.device("cuda" if use_cuda else "cpu")
#     print(f'device: {DEVICE}')

#     # D:\projects\Partial_Spoof_Detection_System\database\train\con_wav\CON_T_0000000.wav
#     # train_data_path=os.path.join(os.getcwd(),'database/train/con_wav')
#     file_name="CON_T_0000000.wav"
#     train_file_path=os.path.join(os.getcwd(),f'database/train/con_wav/{file_name}')
#     print(f"train_file_path= {train_file_path}")
#     import torchaudio
#     waveform, sample_rate = torchaudio.load(train_file_path, normalize=False)
#     print(f"waveform size= {waveform.size()},sample_rate= {sample_rate}")



# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from utils import *
import random
import torchaudio.transforms as T
from torch.utils.data import Dataset
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader , ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.multiprocessing as mp

# class AudiosetDataset(Dataset):
#     def __init__(self, directory,labels_dict,audio_conf=None, transform=True,normalize=True):
#         """
#         Args:
#             directory (str): Path to the directory containing the audio files.
#             labels_dict (dict): Dictionary of labels for each audio file.
#             save_dir (str): Path to the directory where the extracted features will be saved.
#             tokenizer (callable): A tokenizer for preprocessing the audio data.
#             feature_extractor (callable): Feature extractor model (e.g., from HuggingFace).
#             transform (callable, optional): Optional transform to apply to the waveform.
#             normalize (bool, optional): Whether to normalize the waveform. Default is True.
#         """
#         self.directory = directory
#         self.labels_dict = labels_dict
#         self.audio_conf = audio_conf
#         # self.save_dir = save_dir
#         # self.tokenizer = tokenizer
#         # self.feature_extractor = feature_extractor
#         self.transform = transform
#         self.normalize = normalize
#         self.file_list = [f for f in os.listdir(directory) if f.endswith('.wav')]

#         # Ensure the save directory exists
#         # os.makedirs(save_dir, exist_ok=True)

#     def __len__(self):
#         return len(self.file_list)


#     def normalize_waveform(self, waveform):
#         """
#         Normalize the waveform by scaling it to [-1, 1] or applying Z-score normalization.
        
#         Args:
#             waveform (Tensor): The input waveform tensor.
        
#         Returns:
#             Tensor: The normalized waveform.
#         """
#         # Method 1: Normalize to [-1, 1]
#         waveform = waveform / waveform.abs().max()

#         # Method 2: Z-score normalization (mean=0, std=1)
#         # waveform = (waveform - waveform.mean()) / waveform.std()

#         return waveform
    



#     def _wav2fbank(self, file_path):

#         waveform, sr = torchaudio.load(file_path, normalize=False)
#                 # Normalize waveform if needed
#         if self.normalize:
#             waveform = self.normalize_waveform(waveform)

        
#         fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
#                                                 window_type='hanning', num_mel_bins=self.audio_conf.get('num_mel_bins'), dither=0.0, frame_shift=10)

#         target_length = self.audio_conf.get('target_length')
#         n_frames = fbank.shape[0]

#         p = target_length - n_frames

#         # cut and pad
#         if p > 0:
#             m = torch.nn.ZeroPad2d((0, 0, 0, p))
#             fbank = m(fbank)
#         elif p < 0:
#             fbank = fbank[0:target_length, :]


#         return fbank, 0


#     def __getitem__(self, idx):
#         print(f"idx= {idx}")
#         # print(f"idx= {idx[0]}")
#         file_name = self.file_list[idx]
#         file_path = os.path.join(self.directory, file_name)

#         try:
#             # waveform, sample_rate = torchaudio.load(file_path, normalize=True)
#             fbank, mix_lambda=self._wav2fbank(file_path)
#         except Exception as e:
#             print(f"Error loading audio file {file_path}: {e}")
#             return None
        
#         # Apply any other transformations if provided
#         if self.transform:
#             freqm = torchaudio.transforms.FrequencyMasking(self.audio_conf.get('freqm'))
#             timem = torchaudio.transforms.TimeMasking(self.audio_conf.get('timem'))
#             fbank = torch.transpose(fbank, 0, 1)

#         # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
#         fbank = fbank.unsqueeze(0)
#         if self.audio_conf.get('freqm') != 0:
#             fbank = freqm(fbank)
#         if self.audio_conf.get('timem') != 0:
#             fbank = timem(fbank)
#         # squeeze it back, it is just a trick to satisfy new torchaudio version
#         fbank = fbank.squeeze(0)
#         fbank = torch.transpose(fbank, 0, 1)

#         # Normalize waveform if needed
#         if self.normalize:
#             fbank = self.normalize_waveform(fbank)



#         # Return raw waveform and sample rate
#         file_name = file_name.split('.')[0]
#         print(f"file_name= {file_name}")
#         # label = self.labels_dict.get(file_name).astype(int)
#         label = self.labels_dict.get(file_name)
#         # label = torch.tensor(label, dtype=torch.int8)
#         label = torch.tensor(label)

#         return {'fbank': fbank,'label': label, 'file_name': file_name}


import os
import torch
import torchaudio
from torch.utils.data import Dataset


class AudiosetDataset(Dataset):
    def __init__(self, directory, labels_dict, audio_conf=None, transform=True, normalize=True):
        """
        Args:
            directory (str): Path to the directory containing the audio files.
            labels_dict (dict): Dictionary of labels for each audio file.
            audio_conf (dict, optional): Audio configuration dictionary (e.g., num_mel_bins).
            transform (bool, optional): Whether to apply transformations like frequency/time masking.
            normalize (bool, optional): Whether to normalize the waveform. Default is True.
        """
        self.directory = directory
        self.labels_dict = labels_dict
        self.audio_conf = audio_conf
        self.transform = transform
        self.normalize = normalize
        self.file_list = [f for f in os.listdir(directory) if f.endswith('.wav')]

    def __len__(self):
        return len(self.file_list)

    def normalize_waveform(self, waveform):
        """
        Normalize the waveform by scaling it to [-1, 1].
        """
        # print(f"waveform.mean(),waveform.std() = {waveform.mean(),waveform.std()}")
        return waveform / waveform.abs().max()
        # Method 2: Z-score normalization (mean=0, std=1)
        # return (waveform - waveform.mean()) / waveform.std()

    def _wav2fbank(self, file_path):
        """
        Convert a WAV file to its Mel-frequency bank features.
        """
        waveform, sr = torchaudio.load(file_path, normalize=False)
        if self.normalize:
            waveform = self.normalize_waveform(waveform)

        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr,
            use_energy=False, window_type='hanning', num_mel_bins=self.audio_conf.get('num_mel_bins'),
            dither=0.0, frame_shift=10
        )

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]
        p = target_length - n_frames

        # Cut and pad if necessary
        # if p > 0:
        #     fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
        # elif p < 0:
        #     fbank = fbank[:target_length, :]

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]


        return fbank

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.
        """
        if isinstance(idx, tuple):  # Ensure idx is an integer, not a tuple
            raise TypeError(f"Expected integer index, got tuple: {idx}")

        file_name = self.file_list[idx]
        file_path = os.path.join(self.directory, file_name)

        try:
            fbank = self._wav2fbank(file_path)
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None

        # Apply transformations (if specified)
        if self.transform:
            freqm = torchaudio.transforms.FrequencyMasking(self.audio_conf.get('freqm', 0))
            timem = torchaudio.transforms.TimeMasking(self.audio_conf.get('timem', 0))
            fbank = torch.transpose(fbank, 0, 1)  # Adjust for the new torchaudio version
            if self.audio_conf.get('freqm') > 0:
                fbank = freqm(fbank)
            if self.audio_conf.get('timem') > 0:
                fbank = timem(fbank)
            fbank = torch.transpose(fbank, 0, 1)  # Revert transpose

        # Normalize the fbank if required
        if self.normalize:
            fbank = self.normalize_waveform(fbank)

        # Return raw waveform and sample rate
        file_name = file_name.split('.')[0]
        # label = self.labels_dict.get(file_name).astype(int)
        label = self.labels_dict.get(file_name)
        # label = torch.tensor(label, dtype=torch.int8)
        label = torch.tensor(label)

        # print(type(fbank),type(label),type(file_name))
        return {'fbank': fbank, 'label': label, 'file_name': file_name}




# if __name__ == "__main__":

#     # Define the audio configuration
#     audio_conf = {
#         'num_mel_bins': 128,
#         'freqm': 48,  # frequency masking parameter
#         'timem': 192,  # time masking parameter
#         'mixup': 0,  # mix-up rate
#         'dataset': 'Audioset',  # Dataset type
#         'mean': 0.5,  # Mean value for normalization
#         'std': 0.25,  # Standard deviation for normalization
#         'skip_norm': False,  # Do not skip normalization
#         'noise': True,  # Apply noise augmentation
#         'target_length': 1024,  # Target length for spectrogram
#     }

#     # Create an instance of the dataset
#     # label_csv = "path_to_label_csv.csv"
#     # dataset_json_file = "path_to_dataset_json_file.json"
#     # Define training files and labels
#     train_data_path=os.path.join(os.getcwd(),'database/train/con_wav')
#     # train_data_path=os.path.join(os.getcwd(),'database/mini_database/train')
#     train_labels_path=os.path.join(os.getcwd(),'database/utterance_labels/PartialSpoof_LA_cm_train_trl.json')

#     from utils import load_json_dictionary
#     labels_dict= load_json_dictionary(train_labels_path)

#     dataset = AudiosetDataset(train_data_path,labels_dict, audio_conf)

#     sample = dataset[0]  # Correct usage, passing an integer index
#     print(f"fbank size= {sample['fbank'].size()}, label= {sample['label']}")
#     # # Access a data sample
#     for i in range(len(dataset)):
#         sample = dataset[i]
#         if sample['fbank'].size() != torch.Size([1024, 128]):
#             print(f"file_name= {sample['file_name']} fbank size= {sample['fbank'].size()}, label= {sample['label']}")    
#         # print(f"fbank size= {sample['fbank'].size()}, label= {sample['label']}")
#     # fbank: The Mel-frequency spectrogram
#     # label_indices: The label for the sample



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
    
    # if apply_transform:
    #     # Apply pitch shift transform
    #     pitch_shift_transform = PitchShiftTransform(sample_rate=16000, pitch_shift_prob=1.0, pitch_shift_steps=(-2, 2))

    #     # # Initialize the dataset with the transform
    #     augmented_dataset = RawLabeledAudioDataset(
    #         directory=data_path,
    #         labels_dict=labels_dict,
    #         transform=pitch_shift_transform  # Apply pitch shift as part of the dataset transform
    #     )

    #     # Combine datasets
    #     combined_dataset = ConcatDataset([combined_dataset, augmented_dataset])

    # Create the DataLoader
    return DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory,  # Enable page-locked memory for faster data transfer to GPU
        prefetch_factor=prefetch_factor  # How many batches to prefetch per worker
    )
    

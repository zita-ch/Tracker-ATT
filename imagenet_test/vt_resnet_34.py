import argparse
import json
import logging
import os
from pathlib import Path
import sys
import torch, torchvision
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed


import numpy as np
import pandas as pd

from torchvision import models
import time

os.environ['CUDA_VISIBLE_DEVICES'] = "4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FilterTokenizer(nn.Module):
    """
    The Filter Tokenizer extracts visual tokens using point-wise convolutions.
    It takes input of size (HW, C) and outputs a tensor of size (L, D) where:
    - HW : height x width, which represents the number of pixels
    - C : number of input channels
    - L : number of tokens
    - D : number of token channels
    """

    def __init__(self, in_channels: int, token_channels: int, tokens: int) :
        super(FilterTokenizer, self).__init__()

        self.tokens = tokens
        self.in_channels = in_channels
        self.token_channels = token_channels

        self.linear1 = nn.Linear(in_channels, tokens)
        if (in_channels != token_channels):
           self.linear2 = nn.Linear(in_channels, token_channels)
           nn.init.xavier_normal_(self.linear2.weight)
        self.cache1 = None
        self.cache2 = None
        self.token_cache = None

        # initialize weights
        nn.init.xavier_normal_(self.linear1.weight)
        

    def forward(self, x) :
        """
        Expected Input Dimensions: (N, HW, C), where:
        - N: batch size
        - HW: number of pixels
        - C: number of input feature map channels
        
        Expected Output Dimensions: (N, L, D), where:
        - L: number of tokens
        - D: number of token channels
        """

        a = self.linear1(x)  / np.sqrt(self.in_channels)# of size (N, HW, L)
        self.cache1 = a
        a = a.softmax(dim=1)  # softmax for HW dimension, such that every group l features sum to 1
        self.cache2 = a
        a = torch.transpose(a, 1, 2)  # swap dimensions 1 and 2, of size (N, L, HW)
        a = a.matmul(x)  # of size (N, L, C)
        if (self.in_channels != self.token_channels): 
            a = self.linear2(a)  # of size (N, L, D)

        self.token_cache = a
        return a


class RecurrentTokenizer(nn.Module):
    """
    The Recurrent Tokenizer extracts visual tokens by recurrently using tokens generated from
    previous iteration.
    It takes input of size (HW, C), and Tokens matrix of size (L, D) and outputs a tensor 
    of size (L, C) where:
    - HW : height x width, which represents the number of pixels
    - C : number of input feature map channels
    - L : number of tokens
    - D : number of token channels
    """

    def __init__(self, in_channels: int, token_channels: int) :
        super(RecurrentTokenizer, self).__init__()

        self.in_channels = in_channels
        self.token_channels = token_channels

        self.linear1 = nn.Linear(token_channels, token_channels)
        if (in_channels != token_channels):
            self.linear2 = nn.Linear(in_channels, token_channels)
            nn.init.kaiming_normal_(self.linear2.weight)
        self.cache1 = None
        self.cache2 = None
        self.token_cache = None

        # initialize weights
        nn.init.xavier_normal_(self.linear1.weight)
        

    def forward(self, x, t) :
        """
        Expected Input:
        - x : image features input, of size (N, HW, C)
        - t : token features extracted previously, of size (N, L, D)
        
        Expected  Output:
        - t_new : new token features, of size(N, L, D)
        
        where:
        - N : batch size
        - HW: number of pixels
        """

        a = self.linear1(t)  # of size (N, L, D)
        if (self.in_channels != self.token_channels): 
            b = self.linear2(x)  # of size (N, HW, D)
        else:
            b = x
        a = torch.transpose(a, 1, 2)  # transpose by swapping dimensions to become (N, D, L)

        a = b.matmul(a)  # of size (N, HW, L)
        self.cache1 = a
        a = a.softmax(dim=1)  # softmax for HW dimension, such that every group l features sum to 1
        self.cache2 = a
        a = torch.transpose(a, 1, 2)  # transpose by swapping dimensions to become (N, L, HW)
        b = a.matmul(b)  # of size (N, L, D)

        return b
class Projector(nn.Module):
    """
    Projector is a component in the Token-Based Visual Transformer. It is used to
    use the tokens processed by the transformer to refine the feature map pixel
    representation by the information extracted from the visual tokens. It suitable
    in cases where pixel spatial properties need to be preserved, like in segmentation
    It takes the input feature map, of size (HW, C_in), and the output of the transformer
    layer (visual tokens), of size (L, D) and outputs. where:
    - L : number of tokens
    - C_in : number of feature map input channels
    - HW: number of pixels
    - C_out: number of feature map output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int, token_channels: int) :
        super(Projector, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.token_channels = token_channels

        self.cache = None
        
        self.linear1 = nn.Linear(in_channels, token_channels, bias=False) # modifies feature map (query)
        self.linear2 = nn.Linear(token_channels , token_channels, bias=False) # modifies tokens (key)
        self.linear3 = nn.Linear(token_channels, out_channels) # modifies tokens (value)
        
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        nn.init.xavier_normal_(self.linear3.weight)

        #self.norm = nn.BatchNorm1d(out_channels)
 
        # if input size is not same as output size
        # we use downsample to adjust the size of the input feature map
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Linear(in_channels, out_channels),
            )
        
    def forward(self, x, t):
        """
        Expected Input:
        - x : feature map of size (N, HW, C_in)
        - t : visual tokens of size (N, L, D)
        
        Expected Output:
        - x_out : refined feature map of size (N, HW, C_out)
        """    
            
        x_q = self.linear1(x) # of size (N, HW, C_out)
        t_q = self.linear2(t) # of size (N, L, C_out)

        t_q = torch.transpose(t_q, 1, 2) # of size (N, C_out, L) 
        a = x_q.matmul(t_q) # of size (N, HW, L)
        a = a.softmax(dim=2) # of size (N, HW, L)
        self.cache = a
        
        t = self.linear3(t) # of size (N, L, C_out)

        a = a.matmul(t) # of shape (N, HW, C_out)
        
        if self.downsample != None:
            x = self.downsample(x)
            
        x = x + a # of shape (N, HW, C)
        
        #x = torch.transpose(x, 1, 2)
        #x = self.norm(x)
        #x = torch.transpose(x, 1, 2)
        #x = F.relu(x)
        
        return x
class VisualTransformer(nn.Module):
    """
    An implementation of the Token-Based Visual Transformer Module by Wu et al.
    It takes a feature map as input, and depending on whether 
    Parameters:
    - inchannels: number of input channels of feature maps
    - tokens: number of visual tokens to be extracted
    - attn_dim: dimension of projections used in self attention in the transformer
    - tokenization_rounds: number of recurrent iterations for which tokenization is applied
        it includes the first round (filter based) and other rounds (recurrent). 
        (Must be greater than or equal to 1)
    - is_projected: a boolean equal to True with the output is expected to be projected
        back into a spatial map, and False if the output should represent the visual tokens
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        token_channels: int, 
        tokens: int, 
        tokenizer_type: str,
        attn_dim: int, 
        transformer_enc_layers: int,
        transformer_heads: int,
        transformer_fc_dim: int,
        transformer_dropout: int,
        is_projected: bool = True,
    ) -> None:
        super(VisualTransformer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.token_channels = token_channels
        self.attn_dim = attn_dim
        self.is_projected = is_projected
        self.tokens = tokens
        self.tokenizer_type = tokenizer_type
        
        if tokenizer_type not in ['recurrent','filter']:
            raise ValueError('tokenizer type must be either recurrent of filter.')
                        
        self.tokenizer = None
        if tokenizer_type == 'recurrent':
            self.tokenizer = RecurrentTokenizer(in_channels, token_channels)
        else:
            self.tokenizer = FilterTokenizer(in_channels, token_channels, tokens)
        
        
        # self.transformer = SelfAttention(token_channels, token_channels)
        
        #Transformer(token_channels, attn_dim)
        self.transformer = nn.Transformer(
            token_channels, 
            nhead=transformer_heads, 
            num_encoder_layers=transformer_enc_layers, 
            num_decoder_layers=0, 
            dim_feedforward=transformer_fc_dim,
            dropout=transformer_dropout
        )

        # self.transformer = Transformer(
        #     token_channels=token_channels,
        #     attn_dim=attn_dim,
        #     dropout=transformer_dropout
        # )
        
        self.projector = None
        if is_projected:
            self.projector = Projector(in_channels, out_channels, token_channels)
    
    def forward(self, x, t = None) :
        """
        Expected Input:
        - x : input feature maps, of size (N, HW, C)
        
        Expected Output depends on value of self.is_projected.
        If self.is_projected is True:
            - X_out: refined feaure maps, of size (N, HW, C)
        if self.is_projected is False:
            - T_out: visual tokens, of size (N, L, C)
            
        where:
        - N : batch size
        - HW: number of pixels
        - C : number of channels
        - L : number of tokens
        """
        # apply tokenizer
        if self.tokenizer_type == 'filter':
            t = self.tokenizer(x) 
        else:
            t = self.tokenizer(x, t)
        # (N, L, C) -> (L, N, C)
        t = t.permute(1, 0, 2)
        # apply transformer
        t_out = self.transformer(t, t)
        
        t_out = t_out.permute(1, 0, 2) 
        t = t.permute(1, 0, 2)
        
        if self.is_projected:
            out = self.projector(x, t_out)
        
        return out, t        
class VT(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        vt_layers_num: int,
        tokens: int,
        token_channels: int,
        vt_channels: int,
        transformer_enc_layers: int,
        transformer_heads: int,
        transformer_fc_dim: int = 1024,
        transformer_dropout: int = 0.5,
    ) :
        super().__init__()
        
        self.norm_layer = nn.BatchNorm2d
        self.in_channels = in_channels
        self.tokens = tokens
        self.vt_channels = vt_channels
        self.vt_layers_num = vt_layers_num
        # feature map resolution
        self.vt_layer_res = 14
           
        self.bn = nn.BatchNorm2d(self.in_channels)

        self.vt_layers = nn.ModuleList()
        self.vt_layers.append(
            VisualTransformer(
                in_channels=self.in_channels,
                out_channels=self.vt_channels,
                token_channels=token_channels,
                tokens=tokens,
                tokenizer_type='filter',
                attn_dim=token_channels,
                transformer_enc_layers=transformer_enc_layers,
                transformer_heads=transformer_heads,
                transformer_fc_dim=transformer_fc_dim,
                transformer_dropout=transformer_dropout,
                is_projected=True
            )
        )
        
        for _ in range(1, self.vt_layers_num):
            self.vt_layers.append(
                VisualTransformer(
                    in_channels= self.vt_channels,
                    out_channels= self.vt_channels,
                    token_channels=token_channels,
                    tokens=tokens,
                    tokenizer_type='recurrent',
                    attn_dim=token_channels,
                    transformer_enc_layers=transformer_enc_layers,
                    transformer_heads=transformer_heads,
                    transformer_fc_dim=transformer_fc_dim,
                    transformer_dropout=transformer_dropout,
                    is_projected=True
                )
            )
 


    def forward(self, x):
        x = self.bn(x)

        N, C, H, W = x.shape
        
        # flatten pixels
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        
        x, t = self.vt_layers[0](x)
        
        for i in range(1, self.vt_layers_num):
            x, t = self.vt_layers[i](x, t)
    
        
        x = x.permute(0, 2, 1)
        x = x.reshape(N, self.vt_channels, self.vt_layer_res, self.vt_layer_res)
        #x = F.avg_pool2d(x,self.vt_layer_res)
        #x = torch.flatten(x, 1)  
        return x






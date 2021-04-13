import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *


class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads = 1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn , n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val) # Performs linear transformation on data (in features, out features)
        self.fc2 = nn.Linear(dim_val, dim_val)
        
        self.norm1 = nn.LayerNorm(dim_val) # Applies layer normalization over a mini-batch (https://arxiv.org/abs/1607.06450) Improves computational time
        self.norm2 = nn.LayerNorm(dim_val)
    
    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(x + a)

        # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        device = 'cpu'

        elu_tensor = F.elu(self.fc2(x)).to(device)

        a = self.fc1(elu_tensor) #FIXME to check!
        x = self.norm2(x + a)
        
        return x

class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads = 1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        
        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)
        
    def forward(self, x, enc):
        a = self.attn1(x)
        x = self.norm1(a + x)
        
        a = self.attn2(x, kv = enc)
        x = self.norm2(a + x)

        # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
        elu_tensor = F.elu(self.fc2(x)).to(device)
        
        a = self.fc1(elu_tensor) #FIXME to check!
        
        x = self.norm3(x + a)
        return x

class Transformer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_feat_enc, input_feat_dec, seq_len, n_decoder_layers = 1, n_encoder_layers = 1, n_heads = 1):
        super(Transformer, self).__init__()
        #self.dec_seq_len = dec_seq_len
        self.seq_len = seq_len

        # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
        #Initiate encoder and Decoder layers
        self.encs = []
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn, n_heads).to(device))
        
        self.decs = []
        for i in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val, dim_attn, n_heads).to(device))
        
        self.pos = PositionalEncoding(dim_val)
        
        #Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(input_feat_enc, dim_val)
        self.dec_input_fc = nn.Linear(input_feat_dec, dim_val)
        self.out_fc = nn.Linear(self.seq_len * dim_val, input_feat_dec)
    
    def forward(self, x_enc, x_dec):
        #encoder
        first_layer = self.enc_input_fc(x_enc)
        pos_encoder = self.pos(first_layer)
        e = self.encs[0](pos_encoder)
        for enc in self.encs[1:]:
            e = enc(e)
        
        #decoder
        # d = self.decs[0](self.dec_input_fc(x[:,-self.dec_seq_len:]), e)
        d = self.decs[0](self.dec_input_fc(x_dec), e)
        for dec in self.decs[1:]:
            d = dec(d, e)
            
        #output
        x = self.out_fc(d.flatten(start_dim=1))
        
        return x
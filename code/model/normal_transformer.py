# -*- coding: utf-8 -*-
"""
@Time:Created on 2021/7/1
@author: Xu Hangkun
@Filename: model.py
"""

import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=7777):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size,seq length ,embed dim]
            output: [batch size,seq length ,embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)

class GCN(nn.Module):
    def __init__(self,atom_dim,device):
        super().__init__()
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.init_weight()

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self,input,adj,compound_mask = None):
        """
        Args:
            input : atom sequence of compound
            adj : adj matrix of compound
            compound  : compound mask matrix
        Returns:
            output : compound feature
        Shape:
            input : [batch size,compound len,atom dim]
            adj   : [batch size,compound len,compound len]
            compound_mask : [batch size,compound len]
            output: [batch size,compound len,atom dim]
        """
        support = torch.matmul(input, self.weight)
        # support =[batch,num_node,atom_dim]
        if compound_mask is not None:
            adj = adj.masked_fill((compound_mask.unsqueeze(1) == 0).to(self.device),float(0))
        output = torch.bmm(adj, support)
        return output

class NormalTransformerConfig:
    def __init__(
                self,
                protein_dim = 100,
                atom_dim = 34,
                hidden_dim = 256,
                n_layers = 6,
                dropout = 0.1,
                n_heads = 8,
                pf_dim = 512
                ):
        self.protein_dim = protein_dim
        self.atom_dim = atom_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_heads = n_heads
        self.pf_dim = pf_dim

class NormalTransformer(nn.Module):
    def __init__(self,config,device):
        super().__init__()
        self.config = config
        self.device = device
        self.gcn = GCN(self.config.atom_dim,self.device)
        self.comp_ln = nn.Linear(self.config.atom_dim,self.config.hidden_dim)
        self.prot_ln = nn.Linear(self.config.protein_dim,self.config.hidden_dim)
        self.transformer = nn.Transformer(
                d_model = self.config.hidden_dim,
                nhead=self.config.n_heads,
                num_encoder_layers=self.config.n_layers,
                num_decoder_layers=self.config.n_layers,
                dim_feedforward = self.config.pf_dim,
                dropout = self.config.dropout
                )
        self.fc = nn.Sequential(
            nn.Linear(self.config.hidden_dim,1),
            nn.Sigmoid()
        )

    def forward(self,compound,adj,protein,compound_mask=None,protein_mask=None):
        """
        Args:
            compound : compound sequence
            adj      : adj matrix
            protein  : protein matrix
            compoud_mask : mask matrix of compound
            protein_mask : mask matrix of protein
        Return :
            output : probibility of interaction between compound and protein
        Shape:
            compound : [batch size,compound seq len,atom dim]
            protein  : [batch size,protein seq len ,protein dim]
            adj      : [batch size,compound seq len,compound seq len]
            compound mask : [batch size,compound seq len]
            protein mask  : [batch size,compound seq len]
        """
        compound = self.comp_ln(self.gcn(compound,adj,compound_mask))                 # [batch_size,seq_len,hidden_size]
        protein = self.prot_ln(protein)
        compound = compound.permute(1,0,2)
        protein = protein.permute(1,0,2)
        if (compound_mask != None) and (protein_mask != None):
            out = self.transformer(protein,compound,
                    src_key_padding_mask = (protein_mask == 0).to(self.device),
                    tgt_key_padding_mask = (compound_mask == 0).to(self.device)
                    )
            out = out.permute(1,0,2)  #[batch_size,seq_len,hidden_size]
            mean_scale = torch.mean(compound_mask,dim=1).unsqueeze(1).unsqueeze(2).to(self.device)
            out = out*compound_mask.unsqueeze(2).to(self.device)/mean_scale
        else:
            out = self.transformer(protein,compound)
            out = out.permute(1,0,2)
        out = torch.squeeze(torch.mean(out,1))
        return torch.squeeze(self.fc(out))

def test():
    # set the seed
    #torch.manual_seed(7)
    #torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    # define the input
    compound = torch.randn([2,128,34]).to(device)
    compound[:,48:,:] = 1
    compound_mask = torch.ones([2,128])
    compound_mask[:,64:] = 0
    compound_mask.to(device)
    adj = torch.randn([2,128,128]).to(device)
    protein = torch.randn([2,256,100]).to(device)
    protein[:,236:,:] = 0
    protein_mask = torch.ones([2,256])
    protein_mask[:,128:] = 0
    protein_mask.to(device)
    # define model
    config = NormalTransformerConfig()
    model = NormalTransformer(config,device).to(device)
    model.eval()
    pred = model(compound,adj,protein,compound_mask,protein_mask)
    print(pred)

if __name__ == "__main__":
    test()

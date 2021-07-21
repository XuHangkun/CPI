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
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers import PreTrainedModel, AutoModel
from transformers import RobertaConfig

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

class ChemBertMixTransformerConfig:
    def __init__(self,
                protein_dim = 100,
                atom_dim = 34,
                hidden_dim = 64,
                n_layers = 3,
                kerner_size = 7,
                dropout = 0.1,
                n_heads = 8,
                pf_dim = 512,
                chem_bert_pretrain_path = "./ChemBERTa",
                chem_bert_hidden_siz = 768,
                ):
        self.protein_dim = protein_dim
        self.atom_dim = atom_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.kernel_size = kerner_size
        self.dropout = dropout
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.chem_bert_pretrain_path = chem_bert_pretrain_path
        self.chem_bert_hidden_siz = chem_bert_hidden_siz

class ChemBertMixTransformer(nn.Module):
    def __init__(self,config,device):
        super(ChemBertMixTransformer,self).__init__()
        self.config = config
        self.device = device
        self.chem_bert_config = RobertaConfig.from_pretrained(self.config.chem_bert_pretrain_path)
        self.chem_bert = AutoModel.from_config(self.chem_bert_config,add_pooling_layer=False)

        self.comp_ln = nn.Sequential(
            nn.Linear(self.config.chem_bert_hidden_siz,self.config.hidden_dim)
            )
        self.prot_ln = nn.Sequential(
            nn.Linear(self.config.protein_dim,self.config.hidden_dim)
            )

        self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.config.hidden_dim, nhead=self.config.n_heads),
                num_layers = self.config.n_layers
                )
        self.fc = nn.Sequential(
            nn.Linear(self.config.hidden_dim,1),
            nn.Sigmoid()
        )

    def frazing_bert(self):
        print("Frazing ChemBert")
        for param in self.chem_bert.base_model.parameters():
            param.requires_grad = False

    def unfrazing_bert(self):
        print("Unfrazing ChemBert")
        for param in self.chem_bert.base_model.parameters():
            param.requires_grad = True

    def forward(self,compound,protein,compound_mask=None,protein_mask=None):
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
        ### enc_src = [batch size, protein len, hid dim]
        compound = self.chem_bert(
                    input_ids=compound,
                    attention_mask=compound_mask
                    )
        compound = self.comp_ln(compound.last_hidden_state)
        protein = self.prot_ln(protein)
        mix_seq = torch.cat([protein,compound],dim=1)                                 # [batch_size,cmp_seq_len + protein_seq_len,hidden_size]
        mix_seq = mix_seq.permute(1,0,2)
        if compound_mask is not None and protein_mask is not None:
            mix_mask = torch.cat([protein_mask,compound_mask],dim=1).to(self.device)
            out = self.encoder(mix_seq,
                    src_key_padding_mask = (mix_mask == 0)
                    )
            out = out.permute(1,0,2)  #[batch_size,seq_len,hidden_size]
            mean_scale = torch.mean(mix_mask,dim=1).unsqueeze(1).unsqueeze(2)
            out = out*mix_mask.unsqueeze(2)/mean_scale
        else:
            out = self.encoder(mix_seq)
            out = out.permute(1,0,2)
        out = torch.mean(out,1)
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
    compound = torch.ones([2,128]).long()
    compound_mask = torch.ones([2,128])
    compound_mask[:,64:] = 0
    protein = torch.randn([2,256,100])
    protein[:,236:,:] = 0
    protein_mask = torch.ones([2,256])
    protein_mask[:,128:] = 0
    # define model
    config = ChemBertMixTransformerConfig(chem_bert_pretrain_path = "../ChemBERTa")
    model = ChemBertMixTransformer(config,device).to(device)
    #print(model)
    model.eval()
    pred = model(compound.to(device),protein.to(device),compound_mask.to(device),protein_mask.to(device))
    print(pred)

if __name__ == "__main__":
    test()

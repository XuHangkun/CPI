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

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            # print(energy.shape,mask.shape)
            energy = energy.masked_fill(mask == 0, float("-1.e-10"))

        attention = self.dropout_layer(F.softmax(energy, dim=-1))
        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)
        # x = [batch size, n heads, sent len_Q, hid dim // n heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, sent len_Q, n heads, hid dim // n heads]
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        # x = [batch size, src sent len_Q, hid dim]
        x = self.fc(x)
        # x = [batch size, sent len_Q, hid dim]
        return x

class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers,kernel_size , dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        #self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        #pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(self.device)
        #protein = protein + self.pos_embedding(pos)
        #protein = [batch size, protein len,protein_dim]
        conv_input = self.fc(protein)
        # conv_input=[batch size,protein len,hid dim]
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        #conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            #pass through convolutional layer
            #due to the convolution here, padding value will have some effect.
            #use small batch may help
            conved = conv(self.dropout(conv_input))
            #conved = [batch size, 2*hid dim, protein len]

            #pass through GLU activation function
            conved = F.glu(conved, dim=1)
            #conved = [batch size, hid dim, protein len]

            #apply residual connection / high way
            conved = (conved + conv_input) * self.scale
            #conved = [batch size, hid dim, protein len]

            #set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        # conved = [batch size,protein len,hid dim]
        conved = self.ln(conved)
        return conved

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]
        x = x.permute(0, 2, 1)
        # x = [batch size, hid dim, sent len]
        x = self.dropout_layer(F.relu(self.fc_1(x)))
        # x = [batch size, pf dim, sent len]
        x = self.fc_2(x)
        # x = [batch size, hid dim, sent len]
        x = x.permute(0, 2, 1)
        # x = [batch size, sent len, hid dim]
        return x

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
        self.ea = SelfAttention(hid_dim, n_heads, dropout, device)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, compound sent len]
        # src_mask = [batch size, protein len]

        trg = self.ln(trg + self.dropout_layer(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.dropout_layer(self.ea(trg, src, src, src_mask)))
        trg = self.ln(trg + self.dropout_layer(self.pf(trg)))
        return trg

class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.dropout = dropout
        self.device = device
        self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList(
            [DecoderLayer(hid_dim, n_heads, pf_dim,dropout, device) for _ in range(n_layers)]
            )
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, trg, src, trg_mask=None,src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        trg = self.ft(trg)

        # trg = [batch size, compound len, hid dim]

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        # trg = [batch size, compound len, hid dim]
        """Use norm to determine which atom is significant. """
        norm = torch.norm(trg, dim=2)
        # norm = [batch size,compound len]
        norm = F.softmax(norm, dim=1)
        # norm = [batch size,compound len]
        # trg = torch.squeeze(trg,dim=0)
        # norm = torch.squeeze(norm,dim=0)
        sum = torch.zeros((trg.shape[0], self.hid_dim)).to(self.device)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                v = trg[i, j, ]
                v = v * norm[i, j]
                sum[i, ] += v
        # sum = [batch size,hid_dim]
        label = F.relu(self.fc_1(sum))
        label = self.sigmoid(self.fc_2(label))
        return label

class GCN(nn.Module):
    def __init__(self,atom_dim,device):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(atom_dim, atom_dim))
        self.init_weight()
        self.weight.to(device)
        self.device = device

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
        # output = [batch,num_node,atom_dim]
        return output

class TransformerCPIConfig:
    def __init__(
                self,
                protein_dim = 100,
                atom_dim = 34,
                hidden_dim = 64,
                n_layers = 3,
                kerner_size = 7,
                dropout = 0.1,
                n_heads = 8,
                pf_dim = 512
                ):
        self.protein_dim = protein_dim
        self.atom_dim = atom_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.kernel_size = kerner_size
        self.dropout = dropout
        self.n_heads = n_heads
        self.pf_dim = pf_dim

class TransformerCPI(nn.Module):
    def __init__(self,config:TransformerCPIConfig,device):
        super().__init__()
        self.config = config
        self.device = device
        self.gcn = GCN(self.config.atom_dim,device).to(device)
        self.encoder = Encoder(protein_dim = self.config.protein_dim,
                        hid_dim = self.config.hidden_dim,
                        n_layers = self.config.n_layers,
                        kernel_size = self.config.kernel_size,
                        dropout = self.config.dropout,
                        device = device
                        ).to(device)
        self.decoder = Decoder(
            atom_dim = self.config.atom_dim,
            hid_dim = self.config.hidden_dim,
            n_layers = self.config.n_layers,
            n_heads = self.config.n_heads,
            pf_dim = self.config.pf_dim,
            dropout = self.config.dropout,
            device = device
        ).to(device)

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
        enc_src = self.encoder(protein)
        # enc_src = [batch size, protein len, hid dim]
        compound = self.gcn(compound,adj,compound_mask)

        if compound_mask is not None and protein_mask is not None:
            compound_mask = compound_mask.unsqueeze(1).unsqueeze(2).to(self.device)
            # print(compound_mask.shape)
            protein_mask = protein_mask.unsqueeze(1).unsqueeze(2).to(self.device)
            out = self.decoder(compound,enc_src,compound_mask,protein_mask)
        else:
            out = self.decoder(compound,enc_src)
        # out = [batch size, 1]
        return out.squeeze()

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
    config = TransformerCPIConfig()
    model = TransformerCPI(config,device).to(device)
    model.eval()
    pred = model(compound,adj,protein)
    print(pred)

if __name__ == "__main__":
    test()

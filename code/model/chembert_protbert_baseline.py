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
from tape import ProteinBertModel, TAPETokenizer

class ChemBertProtBertBaselineConfig:
    def __init__(self,
                protein_dim = 100,
                atom_dim = 34,
                hidden_dim = 256,
                n_layers = 6,
                dropout = 0.1,
                n_heads = 8,
                pf_dim = 512,
                chem_bert_pretrain_path = "./ChemBERTa",
                chem_bert_hidden_siz = 768,
                prot_bert_pretrain_path = "./protein_bert",
                prot_bert_hidden_siz = 768,
                ):
        self.protein_dim = protein_dim
        self.atom_dim = atom_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.chem_bert_pretrain_path = chem_bert_pretrain_path
        self.chem_bert_hidden_siz = chem_bert_hidden_siz
        self.prot_bert_pretrain_path = prot_bert_pretrain_path
        self.prot_bert_hidden_siz = prot_bert_hidden_siz

class ChemBertProtBertBaseline(nn.Module):
    def __init__(self,config,device):
        super(ChemBertProtBertBaseline,self).__init__()
        self.config = config
        self.device = device
        # load chem bert
        self.chem_bert_config = RobertaConfig.from_pretrained(self.config.chem_bert_pretrain_path)
        self.chem_bert = AutoModel.from_config(self.chem_bert_config,add_pooling_layer=False)
        # load prot bert
        self.prot_bert = ProteinBertModel.from_pretrained(self.config.prot_bert_pretrain_path)
        self.prot_tokenizer = TAPETokenizer(vocab='iupac')

        self.input_dim = self.config.chem_bert_hidden_siz + self.config.prot_bert_hidden_siz
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim,2*self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(2*self.config.hidden_dim,self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim,1),
            nn.Sigmoid()
        )
        self.seen_proteins = {}

    def frazing_bert(self):
        print("Frazing ChemBert and ProtBert")
        for param in self.chem_bert.base_model.parameters():
            param.requires_grad = False

        for param in self.prot_bert.parameters():
            param.requires_grad = False

    def unfrazing_bert(self):
        print("Unfrazing ChemBert and ProtBert")
        for param in self.chem_bert.base_model.parameters():
            param.requires_grad = True

        for param in self.prot_bert.parameters():
            param.requires_grad = True

    def protein_preprocess(self,protein,k=512):
        """
        Args:
            protein : "ABGF..."
        Returns :
            feature : Tensor(feature size)
        """
        if protein in self.seen_proteins:
            return self.seen_proteins[protein]
        p_slices = []
        low_num = 0
        protein_length = len(protein)
        while low_num <= protein_length:
            up_num = min(protein_length,low_num + k)
            p_slices.append(torch.tensor([self.prot_tokenizer.encode(protein[low_num:up_num])]))
            low_num += int(0.8*k)
        p_feature = [torch.mean(self.prot_bert(ids.to(self.device))[0].squeeze(),dim=0) for ids in p_slices]
        p_feature = torch.stack(p_feature)
        p_feature = p_feature.mean(0)
        self.seen_proteins[protein] = p_feature
        return p_feature

    def proteins_process(self,proteins):
        p_features = [self.protein_preprocess(p) for p in proteins]
        return torch.stack(p_features)

    def forward(self,compound,protein,compound_mask=None):
        """
        Args:
            compound : compound sequence
            protein  : protein matrix
            compoud_mask : mask matrix of compound
        Return :
            output : probibility of interaction between compound and protein
        Shape:
            compound : [batch size,compound seq len,atom dim]
            protein  : [batch size,protein seq len ,protein dim]
            compound mask : [batch size,compound seq len]
        """
        # compound [batchsize,seq_len]
        compound = self.chem_bert(
                    input_ids=compound,
                    attention_mask=compound_mask
                    )
        #compound [batchsize,seq_len,feature]
        compound = compound.last_hidden_state.mean(1)

        protein = self.proteins_process(protein)

        out = torch.cat([compound,protein],dim=1)

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
    protein = ["VAYAMDMTLFQAILLDLSMTTCILVYTFIFQWCYDILENR","GCTVEDRCLIGMGAILLNGCVIGSGSLVAAGALITQ"]
    # define model
    config = ChemBertProtBertBaselineConfig(
            chem_bert_pretrain_path = "../ChemBERTa",
            prot_bert_pretrain_path = "../protein_bert"
            )
    model = ChemBertProtBertBaseline(config,device).to(device)
    model.frazing_bert()
    #print(model)
    model.eval()
    pred = model(compound.to(device),protein)
    print(pred)

if __name__ == "__main__":
    test()

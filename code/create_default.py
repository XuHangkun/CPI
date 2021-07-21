import torch
from model import TransformerCPI,TransformerCPIConfig
from model import NormalTransformer,NormalTransformerConfig
from model import MixTransformer,MixTransformerConfig
from model import ChemBertMixTransformer,ChemBertMixTransformerConfig
from model import ChemBertProtBertBaselineConfig,ChemBertProtBertBaseline
from model import Baseline,BaselineConfig
from preprocess import D_PROTEIN_EMBEDDING,D_MOLECULE_EMBEDDING
from preprocess import D_MOLECULE_BASELINE_EMBEDDING, D_PROTEIN_BASELINE_EMBEDDING
from preprocess import CPIDataset
from preprocess import BaselineDataset,ChemBertMixTransformerDataset,ChemBertProtBertBaselineDataset
import numpy as np
from utils import split_train_valid_pnoneside,split_train_valid_compoundoneside

def split_train_valid(data,train_ratio=0.8,type="pnoneside"):
    """split data into two set, train and valid
    Args:
        data : dataframe of data , columns : [smiles,protein,label]
        train_ratio : it's not the ratio of train dataset to total dataset,
                    look into concrete split method for detail information
        type  : the method to split the data , default : pnoneside
    """
    if type == "pnoneside":
        return split_train_valid_pnoneside(data,train_ratio)
    elif type == "cmponeside":
        return split_train_valid_compoundoneside(data,train_ratio)
    else:
        return split_train_valid_pnoneside(data,train_ratio)

def create_model(name="transformercpi",**kwargs):
    """
    Args:
        name : model name, default choice is transformercpi
    Returns:
        model
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    if name.lower() == "normaltransformer":
        config = NormalTransformerConfig(
                atom_dim = D_MOLECULE_EMBEDDING,
                protein_dim = D_PROTEIN_EMBEDDING
                )
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        model = NormalTransformer(config,device)
        print('Create Normal Transformer!!!')
    elif name.lower() == "mixtransformer":
        config = MixTransformerConfig(
                atom_dim = D_MOLECULE_EMBEDDING,
                protein_dim = D_PROTEIN_EMBEDDING
                )
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        model = MixTransformer(config,device)
        print('Create Mix Transformer!!!')
    elif name.lower() == "chembertmixtransformer":
        config = ChemBertMixTransformerConfig(
                atom_dim = D_MOLECULE_EMBEDDING,
                protein_dim = D_PROTEIN_EMBEDDING
                )
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        model = ChemBertMixTransformer(config,device)
        if "frazing_bert" in kwargs.keys():
            if kwargs["frazing_bert"]:
                model.frazing_bert()
            else:
                model.unfrazing_bert()
        else:
            model.frazing_bert()
        print('Create ChemBert Mix Transformer!!!')
    elif name.lower() == "chembertprotbertbaseline":
        config = ChemBertProtBertBaselineConfig(
                atom_dim = D_MOLECULE_EMBEDDING,
                protein_dim = D_PROTEIN_EMBEDDING
                )
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        model = ChemBertProtBertBaseline(config,device)
        model.frazing_bert()
        print('Create ChemBert ProtBert Baseline!!!')
    elif name.lower() == "baseline":
        config = BaselineConfig(
                D_MOLECULE_BASELINE_EMBEDDING,
                D_PROTEIN_BASELINE_EMBEDDING
                )
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        model = Baseline(config)
        print('Create Baseline model!!!')
    else:
        config = TransformerCPIConfig(
                atom_dim = D_MOLECULE_EMBEDDING,
                protein_dim = D_PROTEIN_EMBEDDING
                )
        for key in kwargs.keys():
            setattr(config,key,kwargs[key])
        model = TransformerCPI(config,device)
        print('Create TransformerCPI!!!')
    if config is not None:
        print(config.__dict__)
    return model

def create_dataset(data,model_name,screen=False,protein_max_len=-1,chem_bert_tokenizer=None):
    if model_name.lower() == "baseline":
        return BaselineDataset(data,screen=screen)
    elif model_name.lower() == "chembertmixtransformer":
        return ChemBertMixTransformerDataset(data,chem_bert_tokenizer,screen=screen,protein_max_len = protein_max_len)
    elif model_name.lower() == "chembertprotbertbaseline":
        return ChemBertProtBertBaselineDataset(data,chem_bert_tokenizer,screen=screen,protein_max_len = protein_max_len)
    else:
        return CPIDataset(data,screen=screen,protein_max_len=protein_max_len)

def create_prediction(model,model_name,item,device):
    if model_name.lower() == "baseline":
        molecule,protein,true_y = item
        pred_y = model(molecule.to(device),protein.to(device))
    elif model_name.lower() == "chembertmixtransformer":
        molecule,protein,molecule_mask,protein_mask,true_y = item
        pred_y = model(
            molecule.to(device),protein.to(device),
            molecule_mask.to(device),protein_mask.to(device)
            )
    elif model_name.lower() == "chembertprotbertbaseline":
        molecule,protein,molecule_mask,true_y = item
        pred_y = model(
            molecule.to(device),protein,
            molecule_mask.to(device)
            )
    else:
        molecule,adj,protein,molecule_mask,protein_mask,true_y = item
        pred_y = model(
            molecule.to(device),adj.to(device),protein.to(device),
            molecule_mask.to(device),protein_mask.to(device)
            )
    return pred_y,true_y.to(device)

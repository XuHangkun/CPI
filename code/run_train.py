import os
import tqdm
import torch
import numpy as np
import pandas as pd
import random
import argparse
from torch import optim
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy
from create_default import create_model
from preprocess import D_MOLECULE_EMBEDDING, D_PROTEIN_EMBEDDING
from create_default import create_dataset,create_prediction,split_train_valid
from transformers import RobertaTokenizerFast
from utils import Lookahead
from utils import RAdam
from utils import cal_precision

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int,default=7)
    # parameter of train
    parser.add_argument('--lr', type=float,default=1.e-4)
    parser.add_argument('--weight_decay', type=float,default=1.e-4)
    parser.add_argument('--batch_size', type=int,default=8)
    parser.add_argument('--screen_protein', action="store_true")
    parser.add_argument('--epoch', type=int,default=20)
    parser.add_argument('--save_per_steps', type=int,default=300)
    parser.add_argument('--pretrained_model',default=None)
    parser.add_argument('--pretrained_info',default=None)
    parser.add_argument('--model_name',default="transformercpi")
    parser.add_argument('--model_path',default="../model/baseline_model.pkl")
    parser.add_argument('--train_info_path',default="../model/train_info.csv")
    parser.add_argument('--valid_info_path',default="../model/valid_info.csv")
    parser.add_argument('--label_smoothing',default=0,type=float)
    # parameter of dataset
    parser.add_argument('--input',default="../data/small_data.csv")
    parser.add_argument('--evt_num', type=int,default=1000000)
    parser.add_argument('--split_dataset_method',default = "pnoneside",choices=["pnoneside","cmponeside"])
    parser.add_argument('--save_valid_data_path',default="../model/valid_data.csv")
    # parameter of preprocessing
    parser.add_argument('--protein_max_len',type=int,default=-1)
    # parameter of model
    parser.add_argument('--n_layers',type=int,default=3)
    parser.add_argument('--hidden_dim',type=int,default=64)
    parser.add_argument('--pf_dim',type=int,default=512)
    parser.add_argument('--n_heads',type=int,default=8)
    parser.add_argument('--kerner_size',type=int,default=7)
    parser.add_argument('--dropout',type=float,default=0.1)
    # parameter of ChemBerta
    parser.add_argument('--chemberta_path',default="./ChemBERTa")
    parser.add_argument('--chemberta_tuning',action="store_true")
    args = parser.parse_args()
    print(args)

    # Set Seed
    if 7 is not None:
        print("Set random seed ")
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

    # load tokenizer
    if "chem" in args.model_name.lower():
        chem_tokenizer = RobertaTokenizerFast.from_pretrained(
            args.chemberta_path,max_len=512
        )
    else:
        chem_tokenizer = None

    # Get Train dataset
    all_data = pd.read_csv(args.input)
    all_data.columns = ["smiles","protein","label"]
    all_data = all_data[:args.evt_num]
    # create train and valid dataset
    train_data,valid_data = split_train_valid(all_data,type=args.split_dataset_method)
    valid_data.to_csv(args.save_valid_data_path)
    train_set = create_dataset(train_data,
            args.model_name,screen=args.screen_protein,
            protein_max_len = args.protein_max_len,
            chem_bert_tokenizer = chem_tokenizer
            )
    train_loader = train_set.create_data_loader(batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_set = create_dataset(valid_data,
            args.model_name,screen=args.screen_protein,
            protein_max_len = args.protein_max_len,
            chem_bert_tokenizer = chem_tokenizer
            )
    valid_batch_size = int(args.batch_size*len(valid_data)/len(train_data))
    if valid_batch_size == 0:
        valid_batch_size = 1
    valid_loader = valid_set.create_data_loader(batch_size=valid_batch_size,shuffle=True, drop_last=True)

    # create model
    model = create_model(
            args.model_name,
            frazing_chem_bert = not args.chemberta_tuning,
            n_layers = args.n_layers,
            hidden_dim = args.hidden_dim,
            dropout=args.dropout
            )
    train_info = { "train_loss":[] , "valid_loss":[] , "train_precision":[] , "valid_precision":[]}
    valid_info = {}

    if args.pretrained_model and os.path.exists(args.pretrained_model):
        model.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'))
        if args.pretrained_info and os.path.exists(args.pretrained_info):
            train_info = pd.read_csv(args.pretrained_info).to_dict(orient="list")
        print("Load pretrained model : %s"%(args.pretrained_model))
    model.to(DEVICE)


    # Define optimizer and scheduler
    weight_p , bias_p = [],[]
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    optimizer_inner = RAdam(
        [{'params': weight_p, 'weight_decay': args.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=args.lr)
    optimizer = Lookahead(optimizer_inner, k=5, alpha=0.5)
    #Define optimizer and scheduler
    #optimizer = optim.Adam(model.parameters(),lr=args.lr)
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=3,T_mult=2,eta_min=1.e-6,last_epoch=-1)

    # Training
    step = 0
    for i in range(args.epoch):
        # training
        optimizer.zero_grad()
        with tqdm.tqdm(train_loader, desc=f'Epoch {i + 1}') as epoch_loader:
            for train_item,valid_item in zip(train_loader,valid_loader):
                # train
                model.train()
                pred_y,true_y = create_prediction(model,args.model_name,train_item,DEVICE)
                if args.label_smoothing > 1.e-20:
                    true_y = (1- args.label_smoothing)*true_y.float() + 0.5*args.label_smoothing
                train_loss = binary_cross_entropy(pred_y, true_y.float())
                train_precision = cal_precision(pred_y,true_y)
                train_info["train_loss"].append(train_loss.item())
                train_info["train_precision"].append(train_precision)
                train_loss.backward()
                if (step + 1)%(8*args.batch_size):
                    optimizer.step()
                    optimizer.zero_grad()

                # valid
                model.eval()
                pred_y,true_y = create_prediction(model,args.model_name,valid_item,DEVICE)
                valid_loss = binary_cross_entropy(pred_y, true_y.float())
                valid_precision = cal_precision(pred_y,true_y)
                train_info["valid_loss"].append(valid_loss.item())
                train_info["valid_precision"].append(valid_precision)

                epoch_loader.set_postfix(train_loss=f'{train_loss.item():.4f}',valid_loss=f'{valid_loss.item():.4f}')
                step += 1
                #train_info["lr"].append(scheduler.get_last_lr()[0])
                # save the model
                if step % args.save_per_steps == 0:
                    torch.save(model.state_dict(), args.model_path%(step))
                    print('Baseline model saved to %s'%(args.model_path%(step)))
                    train_info_df = pd.DataFrame(train_info)
                    train_info_df.to_csv(args.train_info_path,index=None)
                    print('Train info saved to %s'%(args.train_info_path))
        # do a prediction after every epoch
        model.eval()
        true_labels = []
        pred_labels = []
        for valid_item in valid_loader:
            pred_y,true_y = create_prediction(model,args.model_name,valid_item,DEVICE)
            true_labels += list(true_y.flatten().cpu().detach().numpy())
            pred_labels += list(pred_y.flatten().cpu().detach().numpy())
        valid_info["epoch_%d_true_y"%(i+1)] = true_labels
        valid_info["epoch_%d_pred_y"%(i+1)] = pred_labels
        valid_info_df = pd.DataFrame(valid_info)
        valid_info_df.to_csv(args.valid_info_path,index=None)
        print('Valid info saved to %s'%(args.valid_info_path))

if __name__ == '__main__':
    main()

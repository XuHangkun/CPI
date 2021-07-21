import torch
from .embeddings import get_molecule_vec, get_protein_vec
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd
from .embeddings import D_MOLECULE_EMBEDDING, D_PROTEIN_EMBEDDING
from tqdm import trange

class CPIDataset(Dataset):
    def __init__(self, df , screen=False, protein_max_len = 256):
        """Dataset of CPI

        Args:
            df : dataframe of data , columns = ["smiles","protein","label"]
            screen : If we screen protein (use for large dataset)
            protein_max_len : max protein len , if protein_max_len < 0, keep the
                                original length
        """
        self.df = df
        self.proteins = {}
        self.samples = []
        self.screen = screen
        self.protein_max_len = protein_max_len
        for i in trange(len(self.df)):
            if self.screen:
                skip = False
                for item in ["B","O","X","J","Z","U"]:
                    if item in self.df["protein"][i]:
                        skip = True
                        break
                if skip:
                    continue
            protein = self.df["protein"][i]
            if protein in self.proteins.keys():
                pass
            else:
                self.proteins[protein] = get_protein_vec(
                        protein,protein_max_len = self.protein_max_len
                        )
            self.samples.append((
                self.df["smiles"][i],
                self.df["protein"][i],
                torch.tensor(self.df["label"][i])
                ))

    @staticmethod
    def collate(samples):
        batch_size = len(samples)
        smiles = [item[0] for item in samples]
        adjs = [item[1] for item in samples]
        proteins = [item[2] for item in samples]
        labels = [item[3] for item in samples]
        compound_max_len = max([item.size(0) for item in smiles])
        protein_max_len = max([item.size(0) for item in proteins])

        compound_mask = torch.zeros((batch_size, compound_max_len))
        protein_mask = torch.zeros((batch_size, protein_max_len))
        adjs_new = torch.zeros((batch_size, compound_max_len, compound_max_len))
        smile = np.zeros((batch_size , compound_max_len, D_MOLECULE_EMBEDDING))
        protein = np.zeros((batch_size , protein_max_len, D_PROTEIN_EMBEDDING))
        for i in range(batch_size):

            seq_len = smiles[i].size(0)
            compound_mask[i, :seq_len] = 1
            smile[i,:seq_len,:] = smiles[i]
            adjs_new[i,:seq_len,:seq_len] = adjs[i] + torch.eye(seq_len)

            seq_len = proteins[i].size(0)
            protein_mask[i, :seq_len] = 1
            protein[i,:seq_len,:] = proteins[i]

        smile = torch.tensor(smile, dtype=torch.float)
        adjs_new = adjs_new.float()
        protein = torch.tensor(protein, dtype=torch.float)

        return (smile,adjs_new,protein,compound_mask,protein_mask,torch.stack(labels))

    def create_data_loader(self, **kwargs):
        return DataLoader(self, collate_fn=CPIDataset.collate, **kwargs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Returns:
            smiles   : shape [compound seq len,atom dim]
            adj      : shape [compound seq len,compound seq len]
            proteins : shape [protein seq len,protein dim]
            label    : shape [1] (1 for true , 0 for false)
        """
        smiles,protein,label = self.samples[index]
        smiles,adj_matrix = get_molecule_vec(smiles)
        protein = self.proteins[protein]
        return (smiles,adj_matrix,protein,label)

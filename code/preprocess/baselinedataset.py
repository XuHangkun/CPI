import torch
from .embeddings import get_baseline_molecule_vec, get_baseline_protein_vec
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd
from tqdm import trange

class BaselineDataset(Dataset):
    def __init__(self, df, screen = False):
        """
        Args:
            df : dataframe of data , columns = ["smiles","protein","label"]
            screen : if we do screen
        """
        self.df = df
        self.proteins = {}
        self.compounds = {}
        self.samples = []
        self.screen = screen
        for i in trange(len(self.df)):
            if self.screen:
                skip = False
                for item in ["B","O","X","J","Z","U"]:
                    if item in self.df["protein"][i]:
                        skip = True
                        break
                if skip:
                    continue
            smiles = self.df["smiles"][i]
            protein = self.df["protein"][i]
            label = self.df["label"][i]
            if protein in self.proteins.keys():
                pass
            else:
                self.proteins[protein] = get_baseline_protein_vec(protein)
            if smiles in self.compounds.keys():
                pass
            else:
                self.compounds[smiles] = get_baseline_molecule_vec(smiles)

            self.samples.append((
                smiles,
                protein,
                torch.tensor([float(label)])
            ))

    @staticmethod
    def collate(samples):
        return map(torch.stack, zip(*samples))

    def create_data_loader(self, **kwargs):
        return DataLoader(self, collate_fn=BaselineDataset.collate, **kwargs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Returns:
            smiles   : shape [D_MOLECULE_EMBEDDING]
            proteins : shape [D_PROTEIN_EMBEDDING]
            label    : shape [1] (1 for true , 0 for false)
        """
        smiles,protein,label = self.samples[index]
        print(self.compounds[smiles],self.proteins[protein])
        return (self.compounds[smiles],self.proteins[protein],label)

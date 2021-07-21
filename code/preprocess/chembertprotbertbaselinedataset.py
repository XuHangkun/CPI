import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd
from tqdm import trange

class ChemBertProtBertBaselineDataset(Dataset):

    def __init__(self, df , molecule_tokenizer ,screen=False, protein_max_len = 256):
        """Dataset of CPI

        Args:
            dfs : DataFrame of smiles and protein sequence, I use dfs here in case you
                want to control ratio of positive and nagetive samples
            molecule_tokenizer : tokenizer of molecule
            screen : If we screen protein (use for large dataset)
            protein_max_len : max protein len , if protein_max_len < 0, keep the
                                original length
        """
        self.df = df
        self.molecule_tokenizer = molecule_tokenizer
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
            self.samples.append((
                self.df["smiles"][i],
                self.df["protein"][i],
                torch.tensor(self.df["label"][i])
                ))

    @staticmethod
    def collate(samples):
        batch_size = len(samples)
        smiles = [item[0] for item in samples]
        proteins = [item[1] for item in samples]
        labels = [item[2] for item in samples]
        compound_max_len = max([item.size(0) for item in smiles])

        compound_mask = torch.zeros((batch_size, compound_max_len))
        smile = 1 + np.zeros((batch_size , compound_max_len))
        for i in range(batch_size):

            seq_len = smiles[i].size(0)
            compound_mask[i, :seq_len] = 1
            smile[i,:seq_len] = smiles[i]

        smile = torch.tensor(smile, dtype=torch.long)
        return (smile,proteins,compound_mask,torch.stack(labels))

    def create_data_loader(self, **kwargs):
        return DataLoader(self, collate_fn=ChemBertProtBertBaselineDataset.collate, **kwargs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        smiles,protein,label = self.samples[index]
        smiles = self.molecule_tokenizer(smiles)["input_ids"]
        smiles = torch.Tensor(smiles).long()
        return (smiles,protein,label)

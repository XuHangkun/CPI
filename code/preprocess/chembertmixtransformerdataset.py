import torch
from .embeddings import get_molecule_vec, get_protein_vec
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd
from .embeddings import D_MOLECULE_EMBEDDING, D_PROTEIN_EMBEDDING
from tqdm import trange

class ChemBertMixTransformerDataset(Dataset):

    def __init__(self, df , molecule_tokenizer ,screen=False, protein_max_len = 256):
        """Dataset of CPI

        Args:
            df : dataframe of data , columns = ["smiles","protein","label"]
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
        self.proteins = {}
        for i in trange(len(self.df)):
            if self.screen:
                skip = False
                for item in ["B","O","X","J","Z","U"]:
                    if item in self.df["protein"][i]:
                        skip = True
                        break
                if skip:
                    continue
            if len(self.df["smiles"][i]) > 512:
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
        proteins = [item[1] for item in samples]
        labels = [item[2] for item in samples]
        compound_max_len = max([item.size(0) for item in smiles])
        protein_max_len = max([item.size(0) for item in proteins])

        compound_mask = torch.zeros((batch_size, compound_max_len))
        protein_mask = torch.zeros((batch_size, protein_max_len))
        smile = 1 + np.zeros((batch_size , compound_max_len))
        protein = np.zeros((batch_size , protein_max_len, D_PROTEIN_EMBEDDING))
        for i in range(batch_size):

            seq_len = smiles[i].size(0)
            compound_mask[i, :seq_len] = 1
            smile[i,:seq_len] = smiles[i]

            seq_len = proteins[i].size(0)
            protein_mask[i, :seq_len] = 1
            protein[i,:seq_len,:] = proteins[i]

        smile = torch.tensor(smile, dtype=torch.long)
        protein = torch.tensor(protein, dtype=torch.float)

        return (smile,protein,compound_mask,protein_mask,torch.stack(labels))

    def create_data_loader(self, **kwargs):
        return DataLoader(self, collate_fn=ChemBertMixTransformerDataset.collate, **kwargs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Returns:
            smiles   : shape [compound seq len]
            proteins : shape [protein seq len,protein dim]
            label    : shape [1] (1 for true , 0 for false)
        """
        smiles,protein,label = self.samples[index]
        smiles = self.molecule_tokenizer(smiles)["input_ids"]
        smiles = torch.Tensor(smiles).long()
        protein = self.proteins[protein]
        return (smiles,protein,label)

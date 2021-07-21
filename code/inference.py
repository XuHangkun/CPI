import logging
import torch
from preprocess import get_molecule_vec, get_protein_vec
from preprocess import get_baseline_molecule_vec, get_baseline_protein_vec
from create_default import create_model
import numpy as np
from transformers import RobertaTokenizerFast
from tape import TAPETokenizer

LOGGER = logging.getLogger(__file__)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# mix transformer ... [you need to change this by your self if you want to use other model]
NORMALTRANSFORMER_MODEL = create_model("mixtransformer",hidden_dim=256,n_layers=1)
NORMALTRANSFORMER_MODEL.load_state_dict(torch.load('../model/mix_transformer_model.pkl', map_location='cpu'))
NORMALTRANSFORMER_MODEL.eval()
NORMALTRANSFORMER_MODEL.to(DEVICE)

@torch.no_grad()
def inference(molecule_smiles: str, protein_fasta: str) -> float:
    LOGGER.debug(f'inference: smiles={molecule_smiles}, fasta={protein_fasta}')
    prob = float('nan')
    try:
        smiles,adj = get_molecule_vec(molecule_smiles)
        smiles = smiles.unsqueeze(0)
        adj = adj.unsqueeze(0)
        prob = NORMALTRANSFORMER_MODEL(
            smiles.to(DEVICE),
            adj.to(DEVICE),
            get_protein_vec(protein_fasta,protein_max_len = 768).unsqueeze(0).to(DEVICE)
        ).item()
    except Exception as e:
        LOGGER.error(f'inference - failed: smiles={molecule_smiles}, fasta={protein_fasta}, error={e}')
    else:
        LOGGER.info(f'inference - success: smiles={molecule_smiles}, fasta={protein_fasta}, prob={prob}')
    return prob

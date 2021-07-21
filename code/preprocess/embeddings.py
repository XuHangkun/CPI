import torch
from rdkit import Chem
from gensim.models import Word2Vec
import numpy as np
import torch.nn.functional as F
import re

"""
This word2vec model is downloaded from https://github.com/lifanchen-simm/transformerCPI.
"""
W2V_MODEL = Word2Vec.load("./preprocess/word2vec_30.model")
D_MOLECULE_EMBEDDING = 34
D_PROTEIN_EMBEDDING = 100

D_MOLECULE_BASELINE_EMBEDDING = 2048
D_PROTEIN_BASELINE_EMBEDDING = 200


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

# Get Molecule and protein feature for transformer models
def atom_features(atom,explicit_H=False,use_chirality=True):
    """Generate atom features including atom symbol(10),degree(7),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2,
                              'other']   # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                  one_of_k_encoding(atom.GetDegree(),degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 10+7+2+6+1=26

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])   # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results

def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return torch.tensor(np.array(adjacency)+np.eye(adjacency.shape[0]),dtype=torch.float)

def get_molecule_vec(smiles):
    """Get sequence represent of compound
    Args:
        smiles : smiles of compound
    Returns:
        atom_feat : sequence represent of compound
        adj_matrix : adjacent matrix of compound
    """
    mol = Chem.MolFromSmiles(smiles)
    # fingerprint = Chem.RDKFingerprint(mol, fpSize=D_MOLECULE_EMBEDDING)
    atom_feat = np.zeros((mol.GetNumAtoms(), D_MOLECULE_EMBEDDING))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    atom_feat = torch.tensor(atom_feat, dtype=torch.float)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat,adj_matrix

def get_protein_vec(protein, k=3 ,protein_max_len = -1):
    """Get the sequence represent of protein
    Args:
        protein : fasta represent protein
        k       : number of protein slice
        protein_max_len : the max length of output protein sequence
    Returns:
        protein_feat : represent of protein sequence
    """
    protein_feat = torch.tensor([list(W2V_MODEL.wv[protein[i:i+k]]) for i in range(len(protein) - k + 1)], dtype=torch.float)
    if protein_max_len > 1:
        # decrease the sequence length by mean pool
        protein_feat = protein_feat.permute(1,0).unsqueeze(0)
        while protein_feat.size(2) > protein_max_len:
            protein_feat = F.avg_pool1d(protein_feat,kernel_size=3, stride=2)
        protein_feat = protein_feat.squeeze(0).permute(1,0)
    return protein_feat

def get_baseline_molecule_vec(smiles):  # length: 2048
    """Get represent of compound
    Args:
        smiles : smiles of compound
    Returns:
        fingerprint : fingerprint compound
    """
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = Chem.RDKFingerprint(mol, fpSize=D_MOLECULE_BASELINE_EMBEDDING)
    return torch.tensor(fingerprint, dtype=torch.float)

def get_baseline_protein_vec(protein, k=3):  # length: 200
    """Get represent of protein
    Args:
        protein : fasta represent protein
        k       : number of protein slice
    Returns:
        protein_feat : represent of protein
    """
    vectors = torch.tensor([list(W2V_MODEL.wv[protein[i:i+k]]) for i in range(len(protein) - k + 1)], dtype=torch.float)
    return torch.cat([vectors.mean(0), vectors.max(0)[0]])  # mean & max

def test():
    protein = "MRGARGAWDFLCVLLLLLR"
    for i in range(200):
        protein += "MRGARGAWDFLCVLLLLLR"
    print(len(protein))
    protein = get_baseline_protein_vec(protein)
    print(protein.shape)
    smiles = "c1ccccc1"
    smiles = get_molecule_vec(smiles)

if __name__ == "__main__":
    test()

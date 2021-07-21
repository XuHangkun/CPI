import torch
from torch import nn

class BaselineConfig:
    def __init__(self,
            d_molecule_embedding = 2048,
            d_protein_embedding = 200,
            hidden_size = 512
            ):
        self.d_molecule_embedding = d_molecule_embedding
        self.d_protein_embedding = d_protein_embedding
        self.hidden_size = hidden_size

class Baseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(self.config.d_molecule_embedding + self.config.d_protein_embedding, self.config.hidden_size),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.config.hidden_size),
            nn.Linear(self.config.hidden_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, molecule_embedding, protein_embedding):
        """
        Args:
            molecule_embedding : [batchsize,d_molecule_embedding]
            protein_embedding  : [batchsize,d_molecule_embedding]
        Returns:
            out : [batchsize,1]
        """
        return self.net(torch.cat([molecule_embedding, protein_embedding], dim=-1))

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
    compound = torch.randn([2,2048]).to(device)
    protein = torch.randn([2,200]).to(device)
    # define model
    config = BaselineConfig(2048,200,256)
    model = Baseline(config).to(device)
    model.eval()
    pred = model(compound,protein)
    print(pred)

if __name__ == "__main__":
    test()

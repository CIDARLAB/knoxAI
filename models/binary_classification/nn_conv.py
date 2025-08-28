import torch.nn.functional as F
from models.base_models import NNConvBase
from utils.ranking_loss import RankingLossWithTies

class NNConvNet(NNConvBase):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, dropout):
        super().__init__(in_channels, hidden_channels, out_channels, edge_dim, dropout)
    
    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    
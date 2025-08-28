import torch.nn.functional as F
from models.base_models import GraphConvBase
from utils.ranking_loss import RankingLossWithTies

class GraphConvRank(GraphConvBase):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, dropout):
        super().__init__(in_channels, hidden_channels, out_channels, edge_dim, dropout)

    def loss(self, pred, label):
        return RankingLossWithTies(margin=1.0)(pred, label)

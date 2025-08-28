import torch.nn.functional as F
from models.base_models import GraphConvBase

class GraphConvNet(GraphConvBase):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, dropout):
        super().__init__(in_channels, hidden_channels, out_channels, edge_dim, dropout)
    
    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    
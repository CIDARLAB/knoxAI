import torch.nn.functional as F
from models.base_models import NNConvBase

class NNConvRegr(NNConvBase):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, dropout):
        super().__init__(in_channels, hidden_channels, out_channels, edge_dim, dropout)
    
    def loss(self, pred, label):
        return F.mse_loss(pred, label)


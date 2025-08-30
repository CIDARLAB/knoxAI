import torch.nn.functional as F
from models.base_models import GraphConvBase

class GraphConvRegr(GraphConvBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def loss(self, pred, label):
        return F.mse_loss(pred, label)

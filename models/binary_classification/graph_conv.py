import torch.nn.functional as F
from models.base_models import GraphConvBase

class GraphConvNet(GraphConvBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)
    
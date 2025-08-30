from models.base_models import NNConvBase
from utils.ranking_loss import RankingLossWithTies

class NNConvRank(NNConvBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def loss(self, pred, label):
        return RankingLossWithTies(margin=1.0)(pred, label)

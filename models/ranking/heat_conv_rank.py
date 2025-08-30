from models.base_models import HEATConvBase
from utils.ranking_loss import RankingLossWithTies

class HEATConvRank(HEATConvBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def loss(self, pred, label):
        return RankingLossWithTies(margin=1.0)(pred, label)
    
import torch.nn.functional as F
from ..utils.ranking_loss import RankingLossWithTies

def get_criterion(task):
    if task == 'regression':
        return F.mse_loss
    elif task == 'binary_classification':
        return F.nll_loss
    elif task == 'ranking':
        return RankingLossWithTies(margin=1.0)
    else:
        raise ValueError(f"Unknown task: {task}")
    
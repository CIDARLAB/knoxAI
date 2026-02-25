import torch.nn.functional as F
from app.utils.ranking_loss import RankingLossWithTies

def get_criterion(task):
    
    if task == 'regression':
        return F.mse_loss, 'MSE'
    
    elif task == 'binary_classification':
        return F.binary_cross_entropy, 'binary_cross_entropy'
    
    elif task == 'multiclass_classification':
        return F.cross_entropy, 'cross_entropy'

    elif task == 'ranking':
        return RankingLossWithTies(margin=1.0), 'ranking'

    else:
        raise ValueError(f"Unknown task: {task}")
    
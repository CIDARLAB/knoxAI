import torch
import torch.nn as nn

class RankingLossWithTies(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, y_pred, y_true):
        """
        y_pred: Tensor of predicted scores, shape (batch_size, num_items)
        y_true: Tensor of integer ranks, shape (batch_size, num_items)
        """
        batch_size, num_items = y_pred.shape
        loss = 0.0
        count = 0

        for b in range(batch_size):
            for i in range(num_items):
                for j in range(num_items):
                    if i == j:
                        continue
                    rank_i = y_true[b, i]
                    rank_j = y_true[b, j]
                    pred_i = y_pred[b, i]
                    pred_j = y_pred[b, j]

                    if rank_i > rank_j:
                        # i should rank higher than j
                        loss += torch.clamp(self.margin - (pred_j - pred_i), min=0)
                        count += 1
                    elif rank_i == rank_j:
                        # i and j are tied, scores should be close
                        loss += (pred_i - pred_j) ** 2
                        count += 1

        return loss / (count + 1e-8)
    
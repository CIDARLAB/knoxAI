# Additional Metrics


def pairwise_accuracy(rank1, rank2):
    n = len(rank1)
    correct_pairs = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Check if the order of the pair (i, j) is the same in both rankings
            if (rank1[i] < rank1[j] and rank2[i] < rank2[j]) or (rank1[i] > rank1[j] and rank2[i] > rank2[j]):
                correct_pairs += 1
            total_pairs += 1

    return correct_pairs / total_pairs

def precision_at_top_k(y_true, y_pred, k):
    """
    y_true: list of true rankings
    y_pred: list of predicted rankings
    k: cutoff rank
    """
    # Get top K predictions
    top_k_preds = y_pred[len(y_pred)-k:]
    
    # Calculate how many of the top K predictions are in the true top K
    relevant_items = len(set(top_k_preds) & set(y_true[len(y_pred)-k:]))
    
    # Precision at K
    return relevant_items / k

def precision_at_bottom_k(y_true, y_pred, k):
    """
    y_true: list of true rankings
    y_pred: list of predicted rankings
    k: cutoff rank
    """
    # Get top K predictions
    bottom_k_preds = y_pred[:k]
    
    # Calculate how many of the top K predictions are in the true top K
    relevant_items = len(set(bottom_k_preds) & set(y_true[:k]))
    
    # Precision at K
    return relevant_items / k

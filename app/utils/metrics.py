import sklearn.metrics as sk_metrics
import scipy.stats as stats

def get_available_metrics():
    available_metrics = {}

    binary_classification_metrics = {
        "f1_score": sk_metrics.f1_score,
        "accuracy": sk_metrics.accuracy_score,
        "precision": sk_metrics.precision_score,
        "recall": sk_metrics.recall_score,
        "roc_auc": sk_metrics.roc_auc_score,
    }

    regression_metrics = {
        "mean_squared_error": sk_metrics.mean_squared_error,
        "mean_absolute_error": sk_metrics.mean_absolute_error,
        "r2_score": sk_metrics.r2_score,
        "pearsonr": stats.pearsonr,
    }

    ranking_metrics = {
        "spearmanr": stats.spearmanr,
        "kendalltau": stats.kendalltau,
        "precision_at_top_k": precision_at_top_k,
        "precision_at_bottom_k": precision_at_bottom_k,
        "pairwise_accuracy": pairwise_accuracy
    }

    available_metrics['binary_classification'] = binary_classification_metrics
    available_metrics['regression'] = regression_metrics
    available_metrics['ranking'] = ranking_metrics

    return available_metrics

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

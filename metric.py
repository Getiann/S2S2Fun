import numpy as np

def AP_by_true_threshold(preds, trues):
    acc_list = []
    for thresh in trues:
        p_pred = preds >= thresh
        p_true = trues >= thresh
        TP = np.sum(p_pred & p_true)
        FP = np.sum(p_pred & (~p_true))
        if TP + FP == 0:
            acc_list.append(0)
        else:
            acc_list.append(TP / (TP + FP))
    return np.mean(acc_list)

def AP_by_pred_threshold(preds, trues):
    acc_list = []
    for thresh_t, thresh_p in zip(trues, preds):
        p_pred = preds >= thresh_p
        p_true = trues >= thresh_t
        TP = np.sum(p_pred & p_true)
        FP = np.sum(p_pred & (~p_true))
        if TP + FP == 0:
            acc_list.append(0)
        else:
            acc_list.append(TP / (TP + FP))
    return np.mean(acc_list)

def pairwise_accuracy(scores, labels, skip_equal=True):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n = len(labels)
    total = 0
    correct = 0
    for i in range(n):
        for j in range(i+1, n):
            if skip_equal and labels[i] == labels[j]:
                continue
            total += 1
            pred = 1 if scores[i] > scores[j] else 0
            true = 1 if labels[i] > labels[j] else 0
            if pred == true:
                correct += 1
    return 0.0 if total == 0 else correct / total

def ndcg_score(y_true, y_pred, k=None):
    """
    Continuous-relevance NDCG.

    y_true: 1D array of real-valued true scores (e.g., absorbance values)
    y_pred: 1D array of predicted scores
    k: compute NDCG@k (if None, use full length)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if k is None:
        k = len(y_true)

    # Sort by predicted ranking (descending)
    order = np.argsort(-y_pred)
    rel = y_true[order][:k]

    # DCG with continuous gain = rel_i
    dcg = np.sum(rel / np.log2(np.arange(2, k + 2)))

    # IDCG: sort by true relevance
    ideal_rel = np.sort(y_true)[::-1][:k]
    idcg = np.sum(ideal_rel / np.log2(np.arange(2, k + 2)))

    if idcg == 0:
        return 0.0

    return dcg / idcg

def pearson_correlation(y_true, y_pred):
    """
    Compute Pearson correlation coefficient between true and predicted scores.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    numerator = np.sum((y_true - mean_true) * (y_pred - mean_pred))
    denominator = np.sqrt(np.sum((y_true - mean_true) ** 2) * np.sum((y_pred - mean_pred) ** 2))

    if denominator == 0:
        return 0.0

    return numerator / denominator

def spearman_correlation(y_true, y_pred):
    """
    Compute Spearman rank correlation coefficient between true and predicted scores.
    """
    from scipy.stats import rankdata

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rank_true = rankdata(y_true)
    rank_pred = rankdata(y_pred)

    return pearson_correlation(rank_true, rank_pred)

def evaluate(y_true, y_pred,k=None,skip_equal=True, pointwise=False):
    results = {}
    if pointwise:
        results['AP_by_threshold'] = AP_by_true_threshold(y_pred, y_true)
    else:
        results['AP_by_threshold'] = AP_by_pred_threshold(y_pred, y_true)
    results['pairwise_accuracy'] = pairwise_accuracy(y_pred, y_true, skip_equal=skip_equal)
    results['ndcg'] = ndcg_score(y_true, y_pred, k)
    results['pearson_correlation'] = pearson_correlation(y_true, y_pred)
    results['spearman_correlation'] = spearman_correlation(y_true, y_pred)
    return results
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import torch


def calc_metrics(labels: torch.Tensor, logits: torch.Tensor):
    prec, rec, f_beta, support = precision_recall_fscore_support(
        labels.numpy(force=True),
        torch.argmax(logits, dim=1).numpy(force=True),
        zero_division=0,
        beta=1.5,
    )

    prec = np.mean(prec) if isinstance(prec, np.ndarray) else float(prec)
    rec = np.mean(rec) if isinstance(rec, np.ndarray) else float(rec)
    f_beta = np.mean(f_beta) if isinstance(f_beta, np.ndarray) else float(f_beta)
    # Calculate ROC-AUC for binary classification
    probabilities = torch.softmax(logits, dim=1).numpy(force=True)
    # NOTE: disabling ROC for now (need to adapt to binary classes)
    # roc_auc = roc_auc_score(labels.numpy(force=True), probabilities[:, 1])
    roc_auc = 0

    return f_beta, prec, rec, roc_auc

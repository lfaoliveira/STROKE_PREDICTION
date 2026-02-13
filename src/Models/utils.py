import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import torch


def calc_metrics(
    labels: torch.Tensor, logits: torch.Tensor, recall_factor: float = 1.8
):
    prec, rec, f_beta, support = precision_recall_fscore_support(
        labels.numpy(force=True),
        torch.argmax(logits, dim=1).numpy(force=True),
        zero_division=0,
        beta=recall_factor,
    )

    prec = np.mean(prec) if isinstance(prec, np.ndarray) else float(prec)
    rec = np.mean(rec) if isinstance(rec, np.ndarray) else float(rec)
    f_beta = np.mean(f_beta) if isinstance(f_beta, np.ndarray) else float(f_beta)

    # NOTE: disabling ROC for now (need to adapt to binary classes)
    # probabilities = torch.softmax(logits, dim=1).numpy(force=True)
    # roc_auc = roc_auc_score(labels.numpy(force=True), probabilities[:, 1])
    roc_auc = 0

    return f_beta, prec, rec, roc_auc


def analyse_test(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    batch_idx: int,
    output_df: pd.DataFrame,
    test_dataset: torch.utils.data.Subset,
):
    data, labels = batch
    logits = model(data)
    labels = torch.squeeze(labels.long())
    predictions = torch.argmax(logits, dim=1)

    # 1. Obter os índices reais do dataset original
    start_idx = batch_idx * data.shape[0]
    end_idx = start_idx + data.shape[0]
    dataset_indices = test_dataset.indices[start_idx:end_idx]

    # 2. Vetorização do Diagnóstico (Muito mais rápido que o loop for)
    # Convertemos para numpy para facilitar a lógica de condições
    preds_np = predictions.numpy(force=True)
    labels_np = labels.numpy(force=True)

    conditions = [
        (preds_np == 1) & (labels_np == 1),  # TP
        (preds_np == 1) & (labels_np == 0),  # FP
        (preds_np == 0) & (labels_np == 1),  # FN
        (preds_np == 0) & (labels_np == 0),  # TN
    ]
    choices = ["TP", "FP", "FN", "TN"]
    results = np.select(conditions, choices, default="ERROR")

    # 3. Set pred and error columns for all indexes of the batch in output_df
    # Build a DataFrame aligned by the original dataset indices so insertion
    # uses index alignment instead of positional assignment
    batch_metrics_df = pd.DataFrame(
        {"pred": preds_np, "error": results}, index=pd.Index(dataset_indices)
    )

    # Update in place using alignment by index and column names
    output_df.update(batch_metrics_df)

    return output_df

import numpy as np
import pandas as pd
import torch
from typing import cast

from DataProcesser.dataset import StrokeDataset


def analyse_test(
    model: torch.nn.Module,
    test_subset: torch.utils.data.Subset[StrokeDataset],
    output_df: pd.DataFrame,
):
    """
    Performs a one-pass analysis logic on the GPU.
    """
    device = next(model.parameters()).device
    model.eval()

    # Move data to GPU for fast inference
    # Note: data and labels are assumed to be Tensors in the underlying dataset
    indices = torch.tensor(test_subset.indices, device=device)

    dataset = cast(StrokeDataset, test_subset.dataset)
    data = dataset.data[indices].to(device)
    labels = dataset.labels[indices].to(device).long().squeeze()

    with torch.no_grad():
        logits = model(data)
        preds = torch.argmax(logits, dim=1)

    # 1. Compute result codes on GPU (TP=1, FP=2, FN=3, TN=4)
    results_code = torch.zeros_like(preds, dtype=torch.uint8)
    results_code[(preds == 1) & (labels == 1)] = 1
    results_code[(preds == 1) & (labels == 0)] = 2
    results_code[(preds == 0) & (labels == 1)] = 3
    results_code[(preds == 0) & (labels == 0)] = 4

    # 2. Convert to CPU for DataFrame assignment
    preds_np = preds.cpu().numpy()
    codes_np = results_code.cpu().numpy()

    # Efficiently map codes to labels
    code_map = {1: "TP", 2: "FP", 3: "FN", 4: "TN", 0: "ERROR"}
    results_str = np.vectorize(code_map.get)(codes_np)

    # 3. Robust Batch assignment using Index Labels (IDs)
    # We use label-based alignment to avoid positional mismatch and key errors.
    output_df = output_df.copy()

    # Get the unique identifiers (IDs) for the samples in the test subset
    # test_subset.indices are offsets into test_subset.dataset.dataframe
    
    target_ids = test_subset.dataset.dataframe.index[test_subset.indices]

    # Initialize columns if they are missing
    for col in ["pred", "error"]:
        if col not in output_df.columns:
            output_df[col] = None

    # Create a results DataFrame indexed by the same IDs to leverage Pandas alignment
    results_to_align = pd.DataFrame(
        {"pred": preds_np, "error": results_str}, index=target_ids
    )

    # Use the Update pattern: it aligns by index label and updates values in-place.
    # This is the safest way to map subset results back to a full dataset.
    if output_df.index.name == target_ids.name:
        output_df.update(results_to_align)
    elif "id" in output_df.columns:
        # Fallback if the index was reset to a default RangeIndex
        output_df = output_df.set_index("id")
        output_df.update(results_to_align)
        output_df = output_df.reset_index()
    else:
        # Final fallback: Positional assignment if IDs cannot be matched via index
        pred_idx = output_df.columns.get_loc("pred")
        error_idx = output_df.columns.get_loc("error")
        output_df.iloc[test_subset.indices, pred_idx] = preds_np
        output_df.iloc[test_subset.indices, error_idx] = results_str

    return output_df, logits, labels

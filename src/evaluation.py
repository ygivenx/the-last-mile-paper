"""
Evaluation metrics for performance status extraction.

Computes precision, recall, and F1 for detection of ECOG, KPS, and Lansky
scores, treating -1 as "not found".
"""

import pandas as pd


def calculate_precision_recall(
    df: pd.DataFrame,
    ground_truth_col: str,
    predicted_col: str,
    metric_name: str,
) -> dict:
    """
    Calculate precision, recall, and F1 for detection (found vs. not found).

    A score of -1 or NaN means "not found". Any other value means "found".

    Args:
        df: DataFrame with ground truth and predicted columns.
        ground_truth_col: Column name for ground truth values.
        predicted_col: Column name for predicted values.
        metric_name: Label for printing (e.g. "ECOG").

    Returns:
        Dict with keys: precision, recall, f1, tp, fp, tn, fn.
    """
    gt = df[ground_truth_col].fillna(-1).astype(float)
    pred = df[predicted_col].fillna(-1).astype(float)

    gt_found = (gt != -1).astype(int)
    pred_found = (pred != -1).astype(int)

    tp = int(((gt_found == 1) & (pred_found == 1)).sum())
    fp = int(((gt_found == 0) & (pred_found == 1)).sum())
    tn = int(((gt_found == 0) & (pred_found == 0)).sum())
    fn = int(((gt_found == 1) & (pred_found == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(f"\n{metric_name} Detection Metrics:")
    print(f"  True Positives (TP):  {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_all_scales(
    matched_df: pd.DataFrame,
) -> dict[str, dict]:
    """
    Run detection metrics for ECOG, KPS, and Lansky.

    Expects columns: ecog_gt, predicted_ecog, kps_gt, predicted_kps,
                     lansky_gt, predicted_lansky.

    Returns:
        Dict keyed by scale name with metric dicts as values.
    """
    results = {}
    for gt_col, pred_col, name in [
        ("ecog_gt", "predicted_ecog", "ECOG"),
        ("kps_gt", "predicted_kps", "KPS"),
        ("lansky_gt", "predicted_lansky", "Lansky"),
    ]:
        results[name.lower()] = calculate_precision_recall(
            matched_df, gt_col, pred_col, name
        )
    return results

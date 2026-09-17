"""
Evaluation metrics for performance status extraction.

Computes precision, recall, and F1 for detection of ECOG, KPS, and Lansky
scores, treating -1 as "not found".

Metrics are *detection* metrics: they ask whether a score for a given scale was
documented and whether the pipeline surfaced it. They do not measure agreement on
the numeric value. The evaluation record is a (reference record x flagged note)
pairing, so a patient-visit holding two reference rows and two flagged notes
contributes four records; see `match_predictions_to_gt` in `run_extraction.py`.
False negatives are captured because ground truth drives the join: every
reference record is evaluated, including those the pipeline flagged nothing for.

All three metrics carry Wilson score intervals: directly for precision and
recall, and for F1 by monotone transformation of a Wilson interval on the
Jaccard index (see `f1_interval`). Every interval is computable from the
confusion matrix alone, so published counts are enough to reproduce them, and
none requires simulation or a prior.

Two caveats travel with every interval here. First, precision and F1 condition
on a denominator that is random rather than fixed by design (the number of
positive predictions, and tp + fp + fn respectively). Second, they assume
independent records, while records are nested within patients. On the benchmark
cohort that assumption survives better than it might: of 725 retained chunks 645
are distinct and they resolve to 557 near-duplicate clusters, so notes from one
patient are largely separate tests of the extraction task. What *is* repetitive
is the notation -- 33 distinct source strings across 134 extractions, with
"ECOG:1" alone accounting for a third -- because the source EHR records
performance status through structured templates. These intervals therefore
describe robustness to contextual variation, not to notational variation, and
should not be extrapolated to narrative or legacy-EHR representations.
"""

import math

import pandas as pd

Z_95 = 1.959963984540054  # two-sided normal quantile for a 95% interval


def wilson_ci(successes: int, trials: int, z: float = Z_95) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    Preferred over the normal approximation here because several cells are at or
    near 0 or 1 (e.g. precision = 1.000), where the normal interval degenerates
    to zero width.

    Returns:
        (lower, upper), or (nan, nan) if trials == 0.
    """
    if trials == 0:
        return (float("nan"), float("nan"))

    p = successes / trials
    denom = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
    return (max(0.0, center - half), min(1.0, center + half))


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


def f1_interval(tp: int, fp: int, tn: int, fn: int) -> tuple[float, float]:
    """
    95% interval for F1, by monotone transformation of a Wilson interval.

    F1 is a strictly increasing function of the Jaccard index. Writing
    m = tp + fp + fn and pi = tp / m,

        F1 = 2*tp / (2*tp + fp + fn) = 2*tp / (tp + m) = 2*pi / (1 + pi)

    and d/dpi [2*pi / (1 + pi)] = 2 / (1 + pi)^2 > 0. So a Wilson interval for
    the binomial proportion pi, with its endpoints pushed through
    pi -> 2*pi / (1 + pi), is an interval for F1 with the same coverage --
    monotone reparameterisation preserves coverage exactly.

    This is preferred over the two obvious alternatives:

    - A nonparametric bootstrap over the record-level outcome vector is WRONG at
      the boundary. When fp = fn = 0 the outcome vector holds only TP and TN
      entries, so every resample reproduces F1 = 1.0 and the interval collapses
      to zero width -- indefensible on 126 records, and that case occurs here
      (e.g. KPS 55/0/71/0).
    - A Dirichlet posterior fixes the degeneracy but introduces a prior, needs
      simulation, carries Monte Carlo error, and can place the observed point
      estimate outside its own interval at very small support.

    The transform approach has none of those problems: deterministic, no prior,
    no simulation, exact at the boundary, and the point estimate is always
    inside the interval. Note tn does not enter -- F1 ignores true negatives by
    construction. It is retained in the signature for interface symmetry.

    Like every interval in this module it conditions on m and assumes
    independent records; see the module docstring on clustering.

    Returns:
        (lower, upper), or (nan, nan) if tp + fp + fn == 0.
    """
    m = tp + fp + fn
    if m == 0:
        return (float("nan"), float("nan"))

    lo, hi = wilson_ci(tp, m)
    return (2 * lo / (1 + lo), 2 * hi / (1 + hi))


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    """
    Derive detection metrics and 95% intervals from a confusion matrix.

    Split out from `calculate_precision_recall` so that already-published
    counts (e.g. the released benchmark confusion matrices) can be turned into
    a metrics table without re-running any model.

    Returns:
        Dict with counts, n, prevalence, precision, recall, f1, and the
        precision_ci / recall_ci / f1_ci pairs.
    """
    n = tp + fp + tn + fn
    n_positive = tp + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_positive if n_positive > 0 else 0.0
    f1 = _f1_from_counts(tp, fp, fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": n,
        "n_positive": n_positive,
        "prevalence": n_positive / n if n > 0 else 0.0,
        "precision": precision,
        "precision_ci": wilson_ci(tp, tp + fp),
        "recall": recall,
        "recall_ci": wilson_ci(tp, n_positive),
        "f1": f1,
        "f1_ci": f1_interval(tp, fp, tn, fn),
    }


def filter_recall_attribution(
    per_model_counts: dict[str, dict],
) -> dict:
    """
    Separate model-attributable from filter-attributable false negatives.

    Every model in the benchmark scored the *same* rule-based filtered input, so
    a reference record that any model recovered proves the filter retained the
    supporting text for that record. The union of true positives across models
    is therefore a lower bound on how many documented values survived the
    filter. Only marginal counts are needed, so the bound is
    `max(tp) <= |union of TPs| <= n_positive`.

    Args:
        per_model_counts: model name -> confusion matrix dict with tp/fn keys,
            all for the same scale and the same reference records.

    Returns:
        Dict with n_positive, retained_at_least, filter_recall_lower_bound, and
        per-model splits of each model's false negatives into the portion
        provably attributable to the model and the residual that cannot be
        attributed from marginal counts alone.
    """
    if not per_model_counts:
        return {}

    n_positive = max(c["tp"] + c["fn"] for c in per_model_counts.values())
    retained_at_least = max(c["tp"] for c in per_model_counts.values())

    per_model = {}
    for name, c in per_model_counts.items():
        # Records another model recovered but this one did not are provably the
        # model's own misses, not the filter's.
        model_attributable = max(0, retained_at_least - c["tp"])
        per_model[name] = {
            "fn": c["fn"],
            "fn_model_attributable_min": model_attributable,
            "fn_unattributable": c["fn"] - model_attributable,
        }

    return {
        "n_positive": n_positive,
        "retained_at_least": retained_at_least,
        "filter_recall_lower_bound": (
            retained_at_least / n_positive if n_positive else float("nan")
        ),
        "per_model": per_model,
    }


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
        Dict with keys: precision, recall, f1, tp, fp, tn, fn, n, n_positive,
        prevalence, and the *_ci interval pairs.
    """
    gt = df[ground_truth_col].fillna(-1).astype(float)
    pred = df[predicted_col].fillna(-1).astype(float)

    gt_found = (gt != -1).astype(int)
    pred_found = (pred != -1).astype(int)

    tp = int(((gt_found == 1) & (pred_found == 1)).sum())
    fp = int(((gt_found == 0) & (pred_found == 1)).sum())
    tn = int(((gt_found == 0) & (pred_found == 0)).sum())
    fn = int(((gt_found == 1) & (pred_found == 0)).sum())

    m = metrics_from_counts(tp, fp, tn, fn)

    print(f"\n{metric_name} Detection Metrics (n={m['n']}, "
          f"documented={m['n_positive']}, prevalence={m['prevalence']:.1%}):")
    print(f"  True Positives (TP):  {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  Precision: {m['precision']:.4f}  "
          f"(95% CI {m['precision_ci'][0]:.4f}-{m['precision_ci'][1]:.4f})")
    print(f"  Recall:    {m['recall']:.4f}  "
          f"(95% CI {m['recall_ci'][0]:.4f}-{m['recall_ci'][1]:.4f})")
    print(f"  F1-Score:  {m['f1']:.4f}  "
          f"(95% CI {m['f1_ci'][0]:.4f}-{m['f1_ci'][1]:.4f})")

    return m


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

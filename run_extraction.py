"""
Standalone performance-status extraction pipeline.

Usage:
    # With OpenAI (default)
    export OPENAI_API_KEY=sk-...
    python run_extraction.py

    # With Anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python run_extraction.py --provider anthropic

    # Custom model
    python run_extraction.py --provider openai --model gpt-4o-mini
"""

import argparse
import ast
from datetime import datetime

import pandas as pd

from src.chunking import chunk_and_filter
from src.evaluation import evaluate_all_scales
from src.extract import run_extraction


def load_notes(path: str) -> pd.DataFrame:
    # MRNs must stay strings: read as integers they lose leading zeros and then
    # silently fail to join against the reference standard.
    return pd.read_csv(path, dtype={"primary_mrn": str})


def load_ground_truth(path: str) -> pd.DataFrame:
    gt = pd.read_csv(path, dtype={"MRN": str})
    gt["Date"] = pd.to_datetime(gt["Date"], format="%m/%d/%y").dt.date
    # Normalize column name (source data has trailing space on "ECOG ")
    gt.columns = [c.strip() for c in gt.columns]
    return gt


GT_SCALES = ("ECOG", "KPS", "Lansky")
PRED_SCALES = ("ecog", "kps", "lansky")


def _surfaced(series: pd.Series) -> pd.Series:
    """Boolean mask of rows carrying an actual score (not -1, not missing)."""
    values = pd.to_numeric(series, errors="coerce")
    return values.notna() & (values != -1)


def _has_source_span(series: pd.Series) -> pd.Series:
    """Boolean mask of rows whose source-span list is non-empty."""

    def nonempty(value) -> bool:
        if isinstance(value, (list, tuple)):
            return len(value) > 0
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return bool(value.strip())
            return len(parsed) > 0 if isinstance(parsed, (list, tuple)) else bool(parsed)
        return False

    return series.apply(nonempty)


def filter_flagged_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the notes for which the pipeline surfaced something.

    A note is retained if any scale carries a value or any scale carries a
    source span. Notes the model read and returned nothing for are dropped
    before the join, and that is what makes the false-negative count meaningful:
    a reference record is charged a miss when *no* retained note for that visit
    surfaced its score, rather than once per silent note. On the benchmark
    cohort this step takes 447 notes to 209 and 143 candidate pairings to 126.
    """
    keep = pd.Series(False, index=pred_df.index)
    for scale in PRED_SCALES:
        if scale in pred_df.columns:
            keep |= _surfaced(pred_df[scale])
        source = f"{scale}_source"
        if source in pred_df.columns:
            keep |= _has_source_span(pred_df[source])

    print(f"  {len(pred_df)} notes -> {int(keep.sum())} surfaced a value or source span")
    return pred_df[keep]


def report_reference_duplicates(gt_df: pd.DataFrame) -> None:
    """
    Report reference visits carrying more than one row, and any that disagree.

    Duplicate rows are *kept*: each is treated as a separate abstraction record
    and scored on its own, which is the unit the published benchmark uses. They
    are reported because duplicate rows that disagree on the same scale are a
    reference-standard inconsistency to resolve upstream, not something to
    silently average away. The benchmark cohort contains one such case.
    """
    key = ["MRN", "Date"]
    duplicated = gt_df.duplicated(subset=key, keep=False)
    if not duplicated.any():
        return

    visits = gt_df.loc[duplicated, key].drop_duplicates()
    print(f"  {len(visits)} reference visits carry duplicate rows "
          f"({int(duplicated.sum())} rows); each row is scored separately")

    for scale in (s for s in GT_SCALES if s in gt_df.columns):
        for visit, group in gt_df[duplicated].groupby(key, sort=False):
            documented = pd.to_numeric(group[scale], errors="coerce")
            documented = documented[documented.notna() & (documented != -1)]
            if documented.nunique() > 1:
                print(f"  WARNING: reference rows disagree on {scale} for {visit}: "
                      f"{sorted(documented.unique())}")


def match_predictions_to_gt(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join note-level predictions onto ground truth on MRN + date.

    The unit of analysis is a (reference record x flagged note) pairing: every
    reference row is paired with every note from that patient-visit that the
    pipeline flagged, and each pairing is scored on its own. A visit holding two
    reference rows and two flagged notes therefore contributes four evaluation
    records. This reproduces the published benchmark denominator (126 records
    from 118 reference rows over 447 notes).

    Ground truth drives the join, so a reference record with no flagged note is
    still evaluated — it joins to nulls and is charged a false negative. That is
    what makes recall a measure of missed documented values rather than a figure
    conditioned on what the model chose to return.
    """
    pred = pred_df.copy()
    pred["service_date"] = pd.to_datetime(
        pred["service_date_key"].astype(str), format="%Y%m%d"
    ).dt.date

    flagged = filter_flagged_predictions(pred)
    report_reference_duplicates(gt_df)

    matched = flagged.merge(
        gt_df,
        left_on=["primary_mrn", "service_date"],
        right_on=["MRN", "Date"],
        how="right",
    )

    matched = matched.rename(columns={
        "ECOG": "ecog_gt",
        "KPS": "kps_gt",
        "Lansky": "lansky_gt",
        "ecog": "predicted_ecog",
        "kps": "predicted_kps",
        "lansky": "predicted_lansky",
    })

    print(f"  {len(gt_df)} reference rows -> {len(matched)} evaluation records")
    return matched


def main():
    parser = argparse.ArgumentParser(
        description="Extract performance status from clinical notes"
    )
    parser.add_argument(
        "--notes", default="data/sample_notes.csv", help="Path to notes CSV"
    )
    parser.add_argument(
        "--ground-truth", default="data/sample_ground_truth.csv",
        help="Path to ground truth CSV"
    )
    parser.add_argument(
        "--provider", default="openai", choices=["openai", "anthropic"],
        help="LLM provider"
    )
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument(
        "--chunk-size", type=int, default=512, help="Chunk size in characters"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=52, help="Chunk overlap in characters"
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save results CSV (default: results/results_<timestamp>.csv)"
    )
    args = parser.parse_args()

    # 1. Load data
    print("Loading notes...")
    notes = load_notes(args.notes)
    print(f"  {len(notes)} notes loaded")

    print("Loading ground truth...")
    gt = load_ground_truth(args.ground_truth)
    print(f"  {len(gt)} ground truth records loaded")

    # 2. Chunk and filter
    print("\nChunking and filtering...")
    chunked = chunk_and_filter(
        notes,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # 3. Extract via LLM
    print(f"\nRunning extraction with {args.provider} ({args.model or 'default'})...")
    predictions = run_extraction(chunked, provider=args.provider, model=args.model)

    # 4. Match and evaluate
    print("\nMatching predictions to ground truth...")
    matched = match_predictions_to_gt(predictions, gt)
    metrics = evaluate_all_scales(matched)

    # 5. Save results
    output_path = args.output
    if not output_path:
        import os
        os.makedirs("results", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/results_{ts}.csv"

    matched.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # 6. Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for scale, m in metrics.items():
        print(f"  {scale.upper():>7s}  F1={m['f1']:.4f}  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}")


if __name__ == "__main__":
    main()

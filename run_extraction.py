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
from datetime import datetime

import pandas as pd

from src.chunking import chunk_and_filter
from src.evaluation import evaluate_all_scales
from src.extract import run_extraction


def load_notes(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_ground_truth(path: str) -> pd.DataFrame:
    gt = pd.read_csv(path)
    gt["Date"] = pd.to_datetime(gt["Date"], format="%m/%d/%y").dt.date
    # Normalize column name (source data has trailing space on "ECOG ")
    gt.columns = [c.strip() for c in gt.columns]
    return gt


def match_predictions_to_gt(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> pd.DataFrame:
    """Right-join predictions to ground truth on MRN + date."""
    pred = pred_df.copy()
    pred["service_date"] = pd.to_datetime(
        pred["service_date_key"].astype(str), format="%Y%m%d"
    ).dt.date

    # Aggregate per (MRN, date) — take first prediction if multiple notes
    pred_agg = (
        pred.groupby(["primary_mrn", "service_date"])
        .first()
        .reset_index()
    )

    matched = gt_df.merge(
        pred_agg,
        left_on=["MRN", "Date"],
        right_on=["primary_mrn", "service_date"],
        how="left",
    )

    matched = matched.rename(columns={
        "ECOG": "ecog_gt",
        "KPS": "kps_gt",
        "Lansky": "lansky_gt",
        "ecog": "predicted_ecog",
        "kps": "predicted_kps",
        "lansky": "predicted_lansky",
    })

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

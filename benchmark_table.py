"""
Rebuild the model-selection benchmark table (Table S1) from released counts.

The bake-off itself ran on institutional PHI and cannot be re-executed outside
MSK, so the per-model confusion matrices are released instead
(`data/benchmark_confusion_matrices.csv`). Everything reported in Table S1 —
precision, recall, F1, their 95% intervals, sample size, class prevalence, and
the split of false negatives into model- vs filter-attributable — is derived
from those counts by this script.

Usage:
    python benchmark_table.py                    # markdown to stdout
    python benchmark_table.py --format csv       # tidy CSV to stdout
    python benchmark_table.py --output results/table_s1.md
"""

import argparse
import sys

import pandas as pd

from src.evaluation import filter_recall_attribution, metrics_from_counts

COUNTS_PATH = "data/benchmark_confusion_matrices.csv"
SCALES = ["ECOG", "KPS", "Lansky"]


def load_counts(path: str = COUNTS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


def build_metrics(counts: pd.DataFrame) -> pd.DataFrame:
    """One row per model x scale, with metrics and intervals."""
    rows = []
    for _, r in counts.iterrows():
        m = metrics_from_counts(int(r.tp), int(r.fp), int(r.tn), int(r.fn))
        rows.append({
            "model": r.model,
            "scale": r.scale,
            "n": m["n"],
            "documented": m["n_positive"],
            "prevalence": m["prevalence"],
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
            "precision": m["precision"],
            "precision_lo": m["precision_ci"][0],
            "precision_hi": m["precision_ci"][1],
            "recall": m["recall"],
            "recall_lo": m["recall_ci"][0],
            "recall_hi": m["recall_ci"][1],
            "f1": m["f1"],
            "f1_lo": m["f1_ci"][0],
            "f1_hi": m["f1_ci"][1],
            "inference_time_sec": r.inference_time_sec,
            "api_errors": r.api_errors,
        })
    return pd.DataFrame(rows)


def _ci(value: float, lo: float, hi: float) -> str:
    return f"{value:.3f} ({lo:.3f}–{hi:.3f})"


def render_markdown(counts: pd.DataFrame, metrics: pd.DataFrame) -> str:
    n_notes = 447
    n_records = int(metrics["n"].iloc[0])
    out = []

    out.append("# Table S1 — Model-selection benchmark\n")
    out.append(
        f"Detection of documented performance status on {n_records} evaluation "
        f"records derived from {n_notes} clinical notes over 11 patients. One "
        "record is a (reference row × flagged note) pairing: 118 reference rows "
        "join to the notes for which the pipeline surfaced a value or a source "
        "span. All models received identical rule-based filtered input, identical "
        "prompts, and identical parsing rules (temperature 0, max_tokens 1192).\n"
    )
    out.append(
        "Precision and recall carry 95% Wilson score intervals; F1 carries a 95% "
        "interval obtained by pushing a Wilson interval on the Jaccard index "
        "through the monotone map π → 2π/(1+π). All three are computable from the "
        "counts alone. They assume independent records and condition on a random "
        "denominator for precision and F1; see the README on what that means "
        "here.\n"
    )

    out.append("## Reference-standard composition\n")
    out.append("| Scale | Documented | Not documented | Prevalence |")
    out.append("|---|---|---|---|")
    for scale in SCALES:
        row = metrics[metrics.scale == scale].iloc[0]
        out.append(
            f"| {scale} | {int(row.documented)} | "
            f"{n_records - int(row.documented)} | {row.prevalence:.1%} |"
        )
    out.append("")

    out.append("## Detection performance (95% CI)\n")
    out.append(
        "| Model | Scale | TP | FP | TN | FN | Precision | Recall | F1 |"
    )
    out.append("|---|---|---|---|---|---|---|---|---|")
    for model in counts.model.unique():
        for scale in SCALES:
            r = metrics[(metrics.model == model) & (metrics.scale == scale)].iloc[0]
            out.append(
                f"| {model} | {scale} | {int(r.tp)} | {int(r.fp)} | {int(r.tn)} | "
                f"{int(r.fn)} | {_ci(r.precision, r.precision_lo, r.precision_hi)} | "
                f"{_ci(r.recall, r.recall_lo, r.recall_hi)} | "
                f"{_ci(r.f1, r.f1_lo, r.f1_hi)} |"
            )
    out.append("")

    out.append("## Throughput and reliability\n")
    out.append(
        "| Model | Job wall-clock (s) | Per-note (s) | Schema/endpoint errors |"
    )
    out.append("|---|---|---|---|")
    for model in counts.model.unique():
        r = counts[counts.model == model].iloc[0]
        out.append(
            f"| {model} | {r.inference_time_sec:.1f} | "
            f"{r.inference_time_sec / n_notes:.3f} | {int(r.api_errors)}/{n_notes} |"
        )
    out.append("")
    out.append(
        "*Job wall-clock* is the elapsed time of one un-warmed Databricks "
        "serverless `ai_query` job across all 447 notes, including the Delta "
        "write; it measures batch throughput, not per-request latency, and is "
        "confounded by endpoint provisioning. *Errors* counts rows for which "
        "`ai_query` (with `failOnError => false`) returned a non-null "
        "`errorMessage` — an endpoint failure or a response that did not "
        "satisfy the JSON schema.\n"
    )

    out.append("## False negatives: model- vs filter-attributable\n")
    out.append(
        "Because every model scored the same filtered input, any record that "
        "*some* model recovered proves the segmentation filter retained the "
        "supporting text for that record. The union of true positives across "
        "models is therefore a lower bound on filter recall.\n"
    )
    out.append(
        "| Scale | Documented | Retained by ≥1 model | Filter recall (lower bound) |"
    )
    out.append("|---|---|---|---|")
    attributions = {}
    for scale in SCALES:
        per_model = {
            r.model: {"tp": int(r.tp), "fn": int(r.fn)}
            for _, r in counts[counts.scale == scale].iterrows()
        }
        a = filter_recall_attribution(per_model)
        attributions[scale] = a
        out.append(
            f"| {scale} | {a['n_positive']} | {a['retained_at_least']} | "
            f"≥ {a['filter_recall_lower_bound']:.1%} |"
        )
    out.append("")

    out.append("| Model | Scale | FN | Provably model's own miss | Unattributable |")
    out.append("|---|---|---|---|---|")
    for model in counts.model.unique():
        for scale in SCALES:
            p = attributions[scale]["per_model"][model]
            out.append(
                f"| {model} | {scale} | {p['fn']} | "
                f"{p['fn_model_attributable_min']} | {p['fn_unattributable']} |"
            )
    out.append("")
    out.append(
        "\"Unattributable\" false negatives are those no model recovered; from "
        "marginal counts alone they may be filter or model misses. They bound "
        "the maximum possible recall loss from segmentation.\n"
    )

    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", default=COUNTS_PATH, help="Path to counts CSV")
    parser.add_argument(
        "--format", default="markdown", choices=["markdown", "csv"],
        help="Output format"
    )
    parser.add_argument("--output", default=None, help="Write to file instead of stdout")
    args = parser.parse_args()

    counts = load_counts(args.counts)
    metrics = build_metrics(counts)

    text = (
        metrics.to_csv(index=False)
        if args.format == "csv"
        else render_markdown(counts, metrics)
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()

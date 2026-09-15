# Performance Status Extraction from Clinical Notes

LLM-based extraction of **ECOG**, **KPS**, and **Lansky** performance status scores from unstructured clinical notes. This repository accompanies our paper on the "last mile" of clinical data — bridging the gap between what clinicians document in free text and the structured data needed for research and clinical trials.

## Overview

Performance status (PS) scores are critical for oncology clinical trials and treatment decisions, yet they are often buried in unstructured clinical notes rather than recorded in structured fields. This project uses large language models to extract PS scores with high precision and recall.

**Pipeline:**
1. **Chunk** clinical notes into overlapping segments (512 chars, 52 overlap)
2. **Filter** chunks by keyword matching (ECOG, KPS, Lansky, Zubrod, WHO PS, etc.)
3. **Extract** structured scores via LLM with a carefully engineered prompt and JSON schema
4. **Evaluate** against manually annotated ground truth

## What the Metrics Mean

All reported metrics are **detection** metrics, not value-agreement metrics. The
unit of analysis is a (patient, visit-date, scale) record. A record is positive
if a score on that scale was documented for that visit; the pipeline is credited
with a true positive if it surfaced that score, and charged a false negative if
it did not. A predicted `-1`, or no surfaced record at all, means "not
documented".

Two consequences worth stating explicitly, because they are easy to misread:

- **False negatives are measured.** Ground truth is joined right-wise, so every
  reference record is evaluated — including visits where the pipeline surfaced
  nothing at all. Recall is therefore a direct measure of missed documented
  values, not a figure conditioned on what the model happened to return.
- **Detection F1 is not comparable to exact-value F1.** Detection asks "was the
  documented score surfaced"; exact-value classification asks "was the numeric
  score right". The same model on the same notes scores differently under the
  two definitions, so any comparison across evaluations must fix the metric
  first.

## Benchmark Results

Model-selection benchmark: **126 ground-truth patient-visit records** derived
from **447 clinical notes** (internal dataset). All four models received
identical rule-based filtered input, identical prompts, and identical parsing
rules (temperature 0, `max_tokens` 1192).

Reference-standard composition — note the low Lansky prevalence, which is why
the Lansky intervals below are wide:

| Scale | Documented | Not documented | Prevalence |
|-------|-----------|----------------|------------|
| ECOG | 102 | 24 | 81.0% |
| KPS | 55 | 71 | 43.7% |
| Lansky | 14 | 112 | 11.1% |

Detection precision, recall, and F1 with 95% intervals (Wilson for precision and
recall, seeded record-level bootstrap for F1):

| Model | Scale | TP | FP | TN | FN | Precision | Recall | F1 |
|-------|-------|----|----|----|----|-----------|--------|-----|
| Claude 3.7 Sonnet | ECOG | 102 | 5 | 19 | 0 | 0.953 (0.895–0.980) | 1.000 (0.964–1.000) | 0.976 (0.953–0.995) |
| Claude 3.7 Sonnet | KPS | 55 | 0 | 71 | 0 | 1.000 (0.935–1.000) | 1.000 (0.935–1.000) | 1.000 (1.000–1.000) |
| Claude 3.7 Sonnet | Lansky | 12 | 0 | 112 | 2 | 1.000 (0.758–1.000) | 0.857 (0.601–0.960) | 0.923 (0.778–1.000) |
| GPT-OSS 120B | ECOG | 102 | 4 | 20 | 0 | 0.962 (0.907–0.985) | 1.000 (0.964–1.000) | 0.981 (0.960–0.995) |
| GPT-OSS 120B | KPS | 46 | 0 | 71 | 9 | 1.000 (0.923–1.000) | 0.836 (0.717–0.911) | 0.911 (0.843–0.962) |
| GPT-OSS 120B | Lansky | 11 | 0 | 112 | 3 | 1.000 (0.741–1.000) | 0.786 (0.524–0.924) | 0.880 (0.700–1.000) |
| GPT-OSS 20B | ECOG | 97 | 3 | 21 | 5 | 0.970 (0.915–0.990) | 0.951 (0.890–0.979) | 0.960 (0.931–0.986) |
| GPT-OSS 20B | KPS | 34 | 1 | 70 | 21 | 0.971 (0.855–0.995) | 0.618 (0.486–0.735) | 0.756 (0.649–0.847) |
| GPT-OSS 20B | Lansky | 11 | 0 | 112 | 3 | 1.000 (0.741–1.000) | 0.786 (0.524–0.924) | 0.880 (0.700–1.000) |
| Llama 3.3 70B | ECOG | 102 | 5 | 19 | 0 | 0.953 (0.895–0.980) | 1.000 (0.964–1.000) | 0.976 (0.953–0.995) |
| Llama 3.3 70B | KPS | 28 | 1 | 70 | 27 | 0.966 (0.828–0.994) | 0.509 (0.381–0.636) | 0.667 (0.540–0.776) |
| Llama 3.3 70B | Lansky | 11 | 0 | 112 | 3 | 1.000 (0.741–1.000) | 0.786 (0.524–0.924) | 0.880 (0.700–1.000) |

Throughput and reliability:

| Model | Job wall-clock (s) | Per-note (s) | Schema/endpoint errors |
|-------|--------------------|--------------|------------------------|
| Claude 3.7 Sonnet | 234.1 | 0.524 | 0/447 |
| GPT-OSS 120B | 32.1 | 0.072 | 0/447 |
| GPT-OSS 20B | 39.4 | 0.088 | 5/447 |
| Llama 3.3 70B | 20.5 | 0.046 | 0/447 |

- **Job wall-clock** is the elapsed time of one un-warmed Databricks serverless
  `ai_query` job over all 447 notes, including the Delta write. It measures batch
  throughput, not per-request latency, and is confounded by endpoint
  provisioning — treat it as an order-of-magnitude cost signal only.
- **Errors** counts rows for which `ai_query` (with `failOnError => false`)
  returned a non-null `errorMessage`: an endpoint failure, or a response that did
  not satisfy the JSON schema.

On this benchmark Claude 3.7 Sonnet is the most accurate model and GPT-OSS 120B
is statistically indistinguishable from it on ECOG while being ~7× faster; the
KPS and Lansky gaps are within overlapping intervals at this sample size. The
production choice of GPT-OSS 120B therefore rests on cost, latency, and
in-environment serving viability, with accuracy as a non-inferiority
constraint rather than the deciding factor.

### Reproducing the benchmark table

The bake-off ran on institutional PHI and cannot be re-executed externally, so
the per-model confusion matrices are released instead. Everything above —
including the intervals and the false-negative attribution — is derived from
those counts:

```bash
python benchmark_table.py                          # markdown
python benchmark_table.py --format csv             # tidy CSV
python benchmark_table.py --output results/table_s1.md
```

### Recall loss attributable to the keyword filter

The filter drops candidate text before the model sees it, so it could in
principle hide documented values. It can be bounded without new experiments:
every model scored the *same* filtered input, so any reference record that some
model recovered proves the filter retained that record's supporting text.

| Scale | Documented | Retained by ≥1 model | Filter recall (lower bound) |
|-------|-----------|----------------------|------------------------------|
| ECOG | 102 | 102 | ≥ 100.0% |
| KPS | 55 | 55 | ≥ 100.0% |
| Lansky | 14 | 12 | ≥ 85.7% |

For ECOG and KPS the filter demonstrably lost nothing: all 9 KPS and all 5 ECOG
false negatives of the smaller models are provably model misses, since a
different model recovered those same records from the same filtered text. Only 2
Lansky records were missed by every model, which bounds the worst-case
segmentation loss for that scale. On this cohort the filter retained 4.0% of
chunks; on the production corpus it forwarded 5.8% of tokens to the model.

## Independently Scored Validation Cohort

The 126-record benchmark above selected the model. Separately, the pipeline's
output over the **126,820-note** validation dataset (Llama 3.3 70B, served on
premises) was scored by the team organizing the bake-off, **against their own
reference standard**, on the records for which they held labels — **12,396** for
ECOG and **12,390** for KPS. Neither the labels nor the scoring code were ours.
This is the largest and the only externally scored evaluation in the paper.

Because their tables report one-vs-rest counts per documented score value and
omit the "not documented" class, they pin down *exact-value* performance
completely but bound *detection* performance only within an interval (see
`independent_validation.py` for the derivation).

Exact-value performance — did the pipeline return the right score?

| Scale | Records | Documented | Micro-F1 | Macro-F1 | Weighted-F1 | Exact agreement among documented |
|-------|---------|-----------|----------|----------|-------------|-----------------------------------|
| ECOG | 12,396 | 9,241 (74.5%) | 0.877 | 0.900 | 0.877 | 80.1% |
| KPS | 12,390 | 4,197 (33.9%) | 0.968 | 0.979 | 0.968 | 97.5% |

Per-score F1 with 95% intervals, for the scores carrying most of the mass:

| Scale | Score | Support | Precision | Recall | F1 |
|-------|-------|---------|-----------|--------|-----|
| ECOG | 0 | 3,750 | 0.978 (0.972–0.983) | 0.781 (0.768–0.794) | 0.869 (0.860–0.877) |
| ECOG | 1 | 4,695 | 0.966 (0.960–0.971) | 0.804 (0.793–0.816) | 0.878 (0.871–0.885) |
| ECOG | 2 | 639 | 0.953 (0.933–0.968) | 0.865 (0.837–0.890) | 0.907 (0.890–0.924) |
| KPS | 70 | 531 | 0.985 (0.971–0.992) | 0.989 (0.976–0.995) | 0.987 (0.980–0.993) |
| KPS | 80 | 1,555 | 0.930 (0.917–0.942) | 0.977 (0.969–0.984) | 0.953 (0.946–0.961) |
| KPS | 90 | 1,706 | 0.982 (0.975–0.987) | 0.968 (0.959–0.976) | 0.975 (0.970–0.980) |

Detection performance — did the pipeline surface a score at all?

| Scale | Detection precision | Detection recall | Detection F1 | Documented values never surfaced |
|-------|---------------------|------------------|--------------|----------------------------------|
| ECOG | 0.970 – 1.000 | 0.801 – 0.825 | 0.877 – 0.904 | 1,615 – 1,843 of 9,241 |
| KPS | 0.962 – 0.986 | 0.975 – 1.000 | 0.968 – 0.993 | 0 – 106 of 4,197 |

The ECOG recall figure is the important one and it is much lower here than the
100% observed on the 126-record benchmark: between 1,615 and 1,843 documented
ECOG values were never surfaced on this cohort. Precision stays high (≥0.97), so
the pipeline is conservative rather than wrong — it abstains rather than
guessing, which is the intended behaviour when an abstention sends the field to
a human. But at this scale ECOG detection recall is roughly 80%, not 100%, and
that is the honest figure for how often the pipeline finds a documented ECOG
score. KPS behaves very differently: near-ceiling detection and 97.5% exact
agreement.

```bash
python independent_validation.py
python independent_validation.py --output results/independent_validation.md
```

## Repository Structure

```
├── data/
│   ├── sample_notes.csv          # Synthetic clinical notes for testing
│   ├── sample_ground_truth.csv   # Corresponding ground truth labels
│   ├── benchmark_confusion_matrices.csv  # Released counts behind Table S1
│   └── independent_validation_confusion_matrices.csv  # Externally scored cohort
├── src/
│   ├── chunking.py               # Text chunking and keyword filtering
│   ├── extract.py                # LLM extraction (OpenAI / Anthropic)
│   ├── evaluation.py             # Detection metrics, CIs, FN attribution
│   └── prompt.py                 # Shared prompt template and JSON schema
├── notebooks/
│   └── databricks_pipeline.py    # Databricks/Spark version (internal pipeline)
├── results/
│   ├── table_s1.md               # Generated benchmark table
│   └── independent_validation.md # Generated external-scoring table
├── benchmark_table.py            # Rebuilds Table S1 from released counts
├── independent_validation.py     # Metrics for the externally scored cohort
├── run_extraction.py             # Standalone entry point
├── CITATION.cff
├── LICENSE
└── pyproject.toml
```

## Quick Start

### 1. Install

```bash
pip install -e ".[all]"
```

### 2. Set your API key

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# OR Anthropic
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run on sample data

```bash
# Using OpenAI (default: gpt-4o)
python run_extraction.py

# Using Anthropic
python run_extraction.py --provider anthropic

# Custom model
python run_extraction.py --provider openai --model gpt-4o-mini
```

Results are saved to `results/results_<timestamp>.csv`.

### Options

```
--notes PATH          Path to notes CSV (default: data/sample_notes.csv)
--ground-truth PATH   Path to ground truth CSV (default: data/sample_ground_truth.csv)
--provider            openai or anthropic (default: openai)
--model               Model name override
--chunk-size          Chunk size in characters (default: 512)
--chunk-overlap       Chunk overlap in characters (default: 52)
--output PATH         Output CSV path
```

## Databricks / Spark Version

The `notebooks/databricks_pipeline.py` file shows the internal Spark-based pipeline using Databricks `ai_query`. This version:
- Uses PySpark UDFs for distributed chunking
- Calls models via Databricks model serving endpoints
- Stores results in Delta tables
- Includes MLflow experiment tracking for multi-model benchmarking

Import it into your Databricks workspace and update the table names / endpoint names for your environment.

## Data Format

**Notes CSV** — must have columns:
- `clinical_note_text_key`: unique note identifier
- `primary_mrn`: patient identifier
- `service_date_key`: date in `YYYYMMDD` format
- `text`: the clinical note text

**Ground Truth CSV** — must have columns:
- `MRN`: patient identifier (matches `primary_mrn`)
- `Date`: date in `M/D/YY` format
- `ECOG`: ground truth ECOG score (-1 if not present)
- `KPS`: ground truth KPS score (-1 if not present)
- `Lansky`: ground truth Lansky score (-1 if not present)

## Prompt Design

The extraction prompt (see `src/prompt.py`) includes:
- Scale definitions (ECOG 0-5, KPS 0-100, Lansky 0-100)
- Allowed mappings (WHO PS → ECOG, Zubrod → ECOG, PPS → KPS)
- Rules for handling ranges, multiple mentions, and qualitative descriptors
- Confidence scoring (1.0 explicit, 0.8 mapped, 0.0 missing/range)
- Few-shot examples covering common extraction scenarios

## Data Availability

The clinical notes and reference labels used in the paper contain protected
health information and cannot be shared. What is released instead is everything
needed to reproduce the reported numbers: the per-model confusion matrices
(`data/benchmark_confusion_matrices.csv`), the externally scored validation
counts (`data/independent_validation_confusion_matrices.csv`), and the scripts
that derive the tables, intervals, and attribution analyses from them
(`benchmark_table.py`, `independent_validation.py`). Synthetic sample data is
provided so the pipeline itself can be run end to end.

## Citation

See `CITATION.cff` for machine-readable metadata. Please cite both the software
and the accompanying article.

<!-- TODO before resubmission: add the Zenodo concept DOI here and in CITATION.cff -->

## License

MIT — see [LICENSE](LICENSE).

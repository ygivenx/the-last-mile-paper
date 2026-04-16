# Performance Status Extraction from Clinical Notes

LLM-based extraction of **ECOG**, **KPS**, and **Lansky** performance status scores from unstructured clinical notes. This repository accompanies our paper on the "last mile" of clinical data — bridging the gap between what clinicians document in free text and the structured data needed for research and clinical trials.

## Overview

Performance status (PS) scores are critical for oncology clinical trials and treatment decisions, yet they are often buried in unstructured clinical notes rather than recorded in structured fields. This project uses large language models to extract PS scores with high precision and recall.

**Pipeline:**
1. **Chunk** clinical notes into overlapping segments (512 chars, 52 overlap)
2. **Filter** chunks by keyword matching (ECOG, KPS, Lansky, Zubrod, WHO PS, etc.)
3. **Extract** structured scores via LLM with a carefully engineered prompt and JSON schema
4. **Evaluate** against manually annotated ground truth

## Benchmark Results

Evaluated on 126 ground truth records across 447 clinical notes (internal dataset):

| Model | ECOG F1 | KPS F1 | Lansky F1 | Inference (s) | Errors |
|-------|---------|--------|-----------|---------------|--------|
| Claude 3.7 Sonnet | 0.9761 | 1.0000 | 0.9231 | 234.1 | 0 |
| GPT-OSS 120B | 0.9808 | 0.9109 | 0.8800 | 32.1 | 0 |
| GPT-OSS 20B | 0.9604 | 0.7556 | 0.8800 | 39.4 | 5 |
| Llama 3.3 70B | 0.9761 | 0.6667 | 0.8800 | 20.5 | 0 |

## Repository Structure

```
├── data/
│   ├── sample_notes.csv          # Synthetic clinical notes for testing
│   └── sample_ground_truth.csv   # Corresponding ground truth labels
├── src/
│   ├── chunking.py               # Text chunking and keyword filtering
│   ├── extract.py                # LLM extraction (OpenAI / Anthropic)
│   ├── evaluation.py             # Precision, recall, F1 metrics
│   └── prompt.py                 # Shared prompt template and JSON schema
├── notebooks/
│   └── databricks_pipeline.py    # Databricks/Spark version (internal pipeline)
├── run_extraction.py             # Standalone entry point
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

## License

MIT

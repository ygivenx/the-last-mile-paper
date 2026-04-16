# Databricks notebook source
# MAGIC %md
# MAGIC # Performance Status Extraction — Databricks/Spark Pipeline
# MAGIC
# MAGIC This notebook shows the internal pipeline used at MSK for extracting ECOG, KPS,
# MAGIC and Lansky performance status scores from clinical notes using Databricks `ai_query`.
# MAGIC
# MAGIC **Requirements:**
# MAGIC - Databricks workspace with serverless or ML runtime
# MAGIC - Access to a model serving endpoint (e.g., `databricks-claude-3-7-sonnet`)
# MAGIC - Delta tables for input notes and ground truth
# MAGIC
# MAGIC **Pipeline overview:**
# MAGIC 1. Load clinical notes from a Delta table
# MAGIC 2. Chunk notes and filter for performance-status keywords
# MAGIC 3. Call an LLM via `ai_query` with structured JSON output
# MAGIC 4. Parse results and evaluate against ground truth

# COMMAND ----------

# MAGIC %pip install langchain-text-splitters

# COMMAND ----------

from pyspark.sql import DataFrame, functions as F, types as T
from pyspark.sql.functions import col, concat, concat_ws, from_json, lit, posexplode
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Chunking and Keyword Filtering

# COMMAND ----------

# ── Text splitter ──
splitter = RecursiveCharacterTextSplitter(
    separators=[r"\s{8}", r"\s{2,}", r""],
    chunk_size=512,
    chunk_overlap=52,
    length_function=len,
    is_separator_regex=True,
)


# Keyword pattern for performance-status mentions
PS_PATTERN = (
    r"\becog\b|\bkps\b|\bzubrod\b|\blansky\b|karnofsky|performance\s+status|"
    r"performance|\bwho\b|physical\s+exam|eastern\s+cooperative\s+oncology\s+group|"
    r"\bpps\b|\bzps\b|functional\s+status|fully\s+active"
)


@F.pandas_udf(ArrayType(StringType()))
def chunk_udf(text_series: pd.Series) -> pd.Series:
    return text_series.apply(splitter.split_text)


def chunk_and_explode(
    df: DataFrame,
    text_col: str = "text",
    text_id_col: str = "clinical_note_text_key",
    patient_id_col: str = "primary_mrn",
    date_col: str = "service_date_key",
) -> DataFrame:
    """Split text into chunks, explode into rows, build chunk_id."""
    exploded = df.withColumn("_chunks", chunk_udf(col(text_col))).select(
        col(text_id_col),
        col(patient_id_col),
        col(date_col),
        posexplode(col("_chunks")).alias("pos", "chunk_text"),
    )
    return exploded.withColumn(
        "chunk_id",
        concat(
            col(text_id_col), lit("_"),
            (col("pos") + 1).cast("string"), lit("_"),
            col(patient_id_col),
        ),
    ).select(text_id_col, patient_id_col, "chunk_id", date_col, "chunk_text")


def add_flag_column(
    df: DataFrame, source_col: str, pattern: str, flag_col: str = "flag",
) -> DataFrame:
    """Add a binary flag column based on regex match."""
    expr = F.lower(F.col(source_col)).rlike(pattern)
    return df.withColumn(
        flag_col,
        F.when(expr, F.lit(1)).otherwise(F.lit(0)).cast(T.IntegerType()),
    )


def combine_flagged_chunks(
    df: DataFrame,
    flag_col: str,
    group_cols: list,
    chunk_id_col: str = "chunk_id",
    chunk_text_col: str = "chunk_text",
) -> DataFrame:
    """Aggregate flagged chunks per group into a single combined text."""
    filtered = df.filter(F.col(flag_col) == 1)
    struct_col = F.struct(
        F.col(chunk_id_col).alias("chunk_id"),
        F.col(chunk_text_col).alias("chunk_text"),
    )
    agg_array = F.sort_array(F.collect_list(struct_col)).alias("chunks_array")
    agg_text = F.concat_ws(
        "\n\n", F.expr("transform(chunks_array, x -> x.chunk_text)")
    ).alias("combined_text")
    agg_count = F.size("chunks_array").alias("chunk_count")

    return filtered.groupBy(*group_cols).agg(agg_array, agg_text, agg_count)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Data and Prepare Chunks
# MAGIC
# MAGIC Replace the table names below with your own Delta tables.

# COMMAND ----------

# ── Load data (replace with your table names) ──
# notes = spark.table("your_catalog.schema.notes_table")
# gt = spark.table("your_catalog.schema.ground_truth_table")

# ── Chunk, flag, and combine ──
# notes_chunks = chunk_and_explode(notes)
# flagged = add_flag_column(notes_chunks, "chunk_text", PS_PATTERN, "ps_flag")
# combined = combine_flagged_chunks(
#     flagged,
#     flag_col="ps_flag",
#     group_cols=["clinical_note_text_key", "service_date_key", "primary_mrn"],
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. LLM Extraction via `ai_query`
# MAGIC
# MAGIC This uses Databricks' `ai_query` SQL function to call a model serving endpoint
# MAGIC with structured JSON output. The prompt and response schema match
# MAGIC `src/prompt.py` in this repository.

# COMMAND ----------

# ── Save combined chunks to a Delta table for the SQL query ──
# combined.write.format("delta").mode("overwrite").option("mergeSchema", "true") \
#     .saveAsTable("your_catalog.schema.notes_chunks")

# ── Run ai_query (swap the endpoint name for your model) ──
AI_QUERY_SQL = """
CREATE OR REPLACE TEMP VIEW raw_perf_stats AS
SELECT
  primary_mrn,
  combined_text,
  clinical_note_text_key,
  service_date_key,
  chunks_array,
  chunk_count,
  current_timestamp() AS prediction_timestamp,
  ai_query(
    '{model_endpoint}',
    concat_ws("",
      'You are a medical AI that extracts structured performance status data from clinical notes.\\n\\n',
      'TASK:\\n',
      'Extract ECOG (0-5), KPS (0-100), and Lansky (0-100) from the note below. ',
      'Use only explicit values or allowed mappings. Work strictly from text. Never guess.\\n\\n',
      'SCALE DEFINITIONS:\\n',
      '1) ECOG: 0=fully active, 5=death.\\n',
      '2) KPS: 0-100 in steps of 10, 100=normal, 0=death.\\n',
      '3) Lansky: 0-100 in steps of 10, 100=fully active, 0=death (pediatric).\\n\\n',
      'ALLOWED MAPPINGS (only these):\\n',
      '- WHO PS (0-4) -> ECOG (WHO 0->ECOG 0 ... WHO 4->ECOG 4)\\n',
      '- Zubrod (0-4) -> ECOG (0->0 ... 4->4)\\n',
      '- PPS (0-100) -> KPS (same numeric value)\\n',
      '- Lansky (0-100) -> Lansky (same numeric value)\\n\\n',
      'RULES:\\n',
      '1) Output must include ALL keys. Defaults if missing: ecog/kps/lansky: -1, *_source: [], *_confidence: 0.0\\n',
      '2) Extract numeric value if clearly stated on target scale or via allowed mapping.\\n',
      '3) If the score is a numeric RANGE (e.g. "ECOG 1-2"), set score=-1 and include range in sources.\\n',
      '4) If single number with descriptive text (e.g. "Lansky 80-active but tired"), extract the number.\\n',
      '5) Multiple mentions: prefer explicit target scale, then latest mention.\\n',
      '6) Do NOT convert directly between ECOG, KPS, and Lansky.\\n',
      '7) Do NOT infer from qualitative descriptors alone without an explicit numeric value.\\n',
      '8) Confidence: 1.0 if explicit, 0.8 if mapped, 0.0 if range or missing.\\n',
      '9) Sources must be verbatim text.\\n',
      '10) Do NOT fabricate values.\\n\\n',
      'Return only a single JSON object. No extra text.\\n\\n',
      '<note>\\n', combined_text, '\\n</note>\\n\\n'
    ),
    responseFormat => '{{
      "type": "json_schema",
      "json_schema": {{
        "name": "performance_status",
        "strict": true,
        "schema": {{
          "type": "object",
          "properties": {{
            "ecog": {{ "type": "integer" }},
            "ecog_source": {{ "type": "array", "items": {{ "type": "string" }} }},
            "ecog_confidence": {{ "type": "number" }},
            "kps": {{ "type": "integer" }},
            "kps_source": {{ "type": "array", "items": {{ "type": "string" }} }},
            "kps_confidence": {{ "type": "number" }},
            "lansky": {{ "type": "integer" }},
            "lansky_source": {{ "type": "array", "items": {{ "type": "string" }} }},
            "lansky_confidence": {{ "type": "number" }}
          }},
          "required": [
            "ecog", "ecog_source", "ecog_confidence",
            "kps", "kps_source", "kps_confidence",
            "lansky", "lansky_source", "lansky_confidence"
          ],
          "additionalProperties": false
        }}
      }}
    }}',
    failOnError => false,
    modelParameters => named_struct('max_tokens', 1192, 'temperature', 0.0)
  ) AS perf_stats_raw
FROM your_catalog.schema.notes_chunks
"""

# spark.sql(AI_QUERY_SQL.format(model_endpoint="databricks-claude-3-7-sonnet"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Parse JSON Results

# COMMAND ----------

PERF_SCHEMA = StructType([
    StructField("ecog", IntegerType()),
    StructField("ecog_source", ArrayType(StringType())),
    StructField("ecog_confidence", StringType()),
    StructField("kps", IntegerType()),
    StructField("kps_source", ArrayType(StringType())),
    StructField("kps_confidence", StringType()),
    StructField("lansky", IntegerType()),
    StructField("lansky_source", ArrayType(StringType())),
    StructField("lansky_confidence", StringType()),
])

# raw_df = spark.table("raw_perf_stats")
# parsed_df = (
#     raw_df.select(
#         "primary_mrn", "clinical_note_text_key", "service_date_key",
#         "chunks_array", "chunk_count", "prediction_timestamp",
#         col("perf_stats_raw.result").alias("json_str"),
#         col("perf_stats_raw.errorMessage").alias("errorStatus"),
#     )
#     .withColumn("perf", from_json(col("json_str"), PERF_SCHEMA))
#     .select(
#         "primary_mrn", "clinical_note_text_key", "service_date_key",
#         "chunks_array", "chunk_count", "prediction_timestamp",
#         "perf.ecog", "perf.ecog_source", "perf.ecog_confidence",
#         "perf.kps", "perf.kps_source", "perf.kps_confidence",
#         "perf.lansky", "perf.lansky_source", "perf.lansky_confidence",
#         "errorStatus",
#     )
# )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evaluate Against Ground Truth

# COMMAND ----------

def calculate_precision_recall(df, ground_truth_col, predicted_col, metric_name):
    """Calculate detection precision/recall treating -1 as 'not found'."""
    df_prep = (
        df.select(
            F.coalesce(F.col(ground_truth_col), F.lit(-1)).alias("ground_truth"),
            F.coalesce(F.col(predicted_col), F.lit(-1)).alias("predicted"),
        )
        .withColumn("gt_found", F.when(F.col("ground_truth") != -1, 1).otherwise(0))
        .withColumn("pred_found", F.when(F.col("predicted") != -1, 1).otherwise(0))
    )
    confusion = df_prep.agg(
        F.sum(F.when((F.col("gt_found") == 1) & (F.col("pred_found") == 1), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("gt_found") == 0) & (F.col("pred_found") == 1), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("gt_found") == 0) & (F.col("pred_found") == 0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col("gt_found") == 1) & (F.col("pred_found") == 0), 1).otherwise(0)).alias("fn"),
    ).collect()[0]

    tp, fp, tn, fn = confusion["tp"] or 0, confusion["fp"] or 0, confusion["tn"] or 0, confusion["fn"] or 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{metric_name} Detection Metrics:")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"  Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Multi-Model Benchmark (Optional)
# MAGIC
# MAGIC Loop over multiple model endpoints and log results to MLflow.

# COMMAND ----------

# import mlflow
#
# MODELS = [
#     {"name": "Claude 3.7 Sonnet",  "endpoint": "databricks-claude-3-7-sonnet"},
#     {"name": "Llama 3.3 70B",      "endpoint": "databricks-meta-llama-3-3-70b-instruct"},
#     {"name": "GPT-OSS 20B",        "endpoint": "databricks-gpt-oss-20b"},
#     {"name": "GPT-OSS 120B",       "endpoint": "databricks-gpt-oss-120b"},
# ]
#
# mlflow.set_experiment("/Users/you@org.com/perf-status-benchmark")
#
# for model in MODELS:
#     metrics, inf_time, errors = run_model_and_evaluate(model["endpoint"])
#     with mlflow.start_run(run_name=model["name"]):
#         mlflow.log_params({"model_endpoint": model["endpoint"], "chunk_size": 512})
#         mlflow.log_metric("inference_time_sec", inf_time)
#         for scale, m in metrics.items():
#             for k, v in m.items():
#                 mlflow.log_metric(f"{scale}_{k}", v)

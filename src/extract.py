"""
Standalone performance-status extraction using OpenAI or Anthropic APIs.

No Spark/Databricks dependencies. Works with pandas DataFrames.
"""

import json
import os
import time
from typing import Literal

import pandas as pd

from .prompt import EXTRACTION_PROMPT, RESPONSE_SCHEMA, SYSTEM_PROMPT


def _call_openai(text: str, model: str) -> dict:
    from openai import OpenAI

    client = OpenAI()  # uses OPENAI_API_KEY env var
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=1192,
        response_format=RESPONSE_SCHEMA,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EXTRACTION_PROMPT.format(note_text=text)},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def _call_anthropic(text: str, model: str) -> dict:
    import anthropic

    client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
    # Anthropic doesn't support json_schema response_format the same way;
    # we instruct via prompt and parse the JSON output.
    resp = client.messages.create(
        model=model,
        max_tokens=1192,
        temperature=0.0,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": EXTRACTION_PROMPT.format(note_text=text)},
        ],
    )
    return json.loads(resp.content[0].text)


def extract_performance_status(
    text: str,
    provider: Literal["openai", "anthropic"] = "openai",
    model: str | None = None,
) -> dict:
    """
    Extract ECOG, KPS, and Lansky scores from a clinical note.

    Args:
        text: Combined chunk text from a clinical note.
        provider: "openai" or "anthropic".
        model: Model name override. Defaults to gpt-4o / claude-sonnet-4-20250514.

    Returns:
        Parsed dict with ecog, kps, lansky scores, sources, and confidences.
    """
    defaults = {"openai": "gpt-4o", "anthropic": "claude-sonnet-4-20250514"}
    model = model or defaults[provider]

    if provider == "openai":
        return _call_openai(text, model)
    elif provider == "anthropic":
        return _call_anthropic(text, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def run_extraction(
    chunked_df: pd.DataFrame,
    provider: Literal["openai", "anthropic"] = "openai",
    model: str | None = None,
    text_col: str = "combined_text",
) -> pd.DataFrame:
    """
    Run extraction on all rows of a chunked DataFrame.

    Adds columns: ecog, ecog_source, ecog_confidence, kps, kps_source,
                  kps_confidence, lansky, lansky_source, lansky_confidence,
                  error_message.

    Returns:
        Copy of input DataFrame with extraction result columns appended.
    """
    results = []
    start = time.time()

    for idx, row in chunked_df.iterrows():
        try:
            parsed = extract_performance_status(row[text_col], provider, model)
            parsed["error_message"] = None
        except Exception as e:
            parsed = {
                "ecog": None, "ecog_source": [], "ecog_confidence": 0.0,
                "kps": None, "kps_source": [], "kps_confidence": 0.0,
                "lansky": None, "lansky_source": [], "lansky_confidence": 0.0,
                "error_message": str(e),
            }
        results.append(parsed)

    elapsed = time.time() - start
    print(f"Extraction complete: {len(results)} notes in {elapsed:.1f}s")

    result_df = pd.DataFrame(results)
    return pd.concat(
        [chunked_df.reset_index(drop=True), result_df.reset_index(drop=True)],
        axis=1,
    )

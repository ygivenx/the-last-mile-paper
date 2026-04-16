"""
Text chunking and keyword filtering for clinical notes.

Splits long notes into overlapping chunks and flags those containing
performance-status keywords so only relevant text is sent to the LLM.
"""

import re

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Regex pattern matching performance-status keywords
PS_PATTERN = re.compile(
    r"\becog\b|\bkps\b|\bzubrod\b|\blansky\b|karnofsky|performance\s+status|"
    r"performance|\bwho\b|physical\s+exam|eastern\s+cooperative\s+oncology\s+group|"
    r"\bpps\b|\bzps\b|functional\s+status|fully\s+active",
    re.IGNORECASE,
)

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 52


def build_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=[r"\s{8}", r"\s{2,}", r""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=True,
    )


def chunk_and_filter(
    notes_df: pd.DataFrame,
    text_col: str = "text",
    note_id_col: str = "clinical_note_text_key",
    patient_id_col: str = "primary_mrn",
    date_col: str = "service_date_key",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> pd.DataFrame:
    """
    Split notes into chunks, keep only those matching PS_PATTERN,
    and recombine per (note, patient, date).

    Returns a DataFrame with columns:
        note_id_col, patient_id_col, date_col, combined_text, chunk_count
    """
    splitter = build_splitter(chunk_size, chunk_overlap)
    rows = []

    for _, row in notes_df.iterrows():
        text = str(row[text_col])
        chunks = splitter.split_text(text)
        flagged = [c for c in chunks if PS_PATTERN.search(c)]

        if flagged:
            rows.append(
                {
                    note_id_col: row[note_id_col],
                    patient_id_col: row[patient_id_col],
                    date_col: row[date_col],
                    "combined_text": "\n\n".join(flagged),
                    "chunk_count": len(flagged),
                }
            )

    total = len(notes_df)
    flagged_count = len(rows)
    print(
        f"Chunking complete: {flagged_count}/{total} notes "
        f"({100 * flagged_count / total:.1f}%) have PS-relevant chunks"
    )
    return pd.DataFrame(rows)

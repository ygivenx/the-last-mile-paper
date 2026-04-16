"""
Shared prompt template for performance status extraction.

Used by both the standalone and Databricks versions to ensure consistency.
"""

SYSTEM_PROMPT = (
    "You are a medical AI that extracts structured performance status data "
    "from clinical notes."
)

EXTRACTION_PROMPT = """\
TASK:
Extract ECOG (0-5), KPS (0-100), and Lansky (0-100) from the note below. \
Use only explicit values or allowed mappings. Work strictly from text. Never guess.

SCALE DEFINITIONS:
1) ECOG: 0=fully active, 5=death.
2) KPS: 0-100 in steps of 10, 100=normal, 0=death.
3) Lansky: 0-100 in steps of 10, 100=fully active, 0=death (pediatric).

ALLOWED MAPPINGS (only these):
- WHO PS (0-4) -> ECOG (WHO 0->ECOG 0 ... WHO 4->ECOG 4)
- Zubrod (0-4) -> ECOG (0->0 ... 4->4)
- PPS (0-100) -> KPS (same numeric value)
- Lansky (0-100) -> Lansky (same numeric value)

RULES:
1) Output must include ALL keys. Defaults if missing:
   - ecog/kps/lansky: -1
   - *_source: []
   - *_confidence: 0.0
2) Extract numeric value if clearly stated on target scale or via allowed mapping, \
even if accompanied by descriptive text.
3) If the score is a numeric RANGE with hyphen/dash (e.g., "ECOG 1-2" or "KPS 90-100" \
or "Lansky 80-90"), set score=-1 and include the range verbatim in sources.
4) If the score is a single number with descriptive text (e.g., "Lansky 80-active but \
tired" or "ECOG 2 ambulatory"), extract the number and set confidence=1.0.
5) Multiple mentions for the same endpoint:
   - Prefer explicit target scale over mapped.
   - Prefer the latest mention if chronology is clear.
   - If unresolved, set score=-1 and include all relevant snippets in sources.
6) Do NOT convert directly between ECOG, KPS, and Lansky.
7) Do NOT infer numbers from qualitative descriptors alone ("good", "poor", "excellent") \
without an explicit numeric value.
8) Confidence:
   - 1.0 if target scale and numeric value explicitly stated (with or without descriptive text).
   - 0.8 if from allowed mapping only.
   - 0.0 if no numeric value or only a range.
9) Sources must be verbatim text supporting the value or range for each endpoint.
10) If the note lacks needed information, use the defaults and sources above. Do NOT fabricate values.

OUTPUT:
Return only a single JSON object matching the schema exactly. No extra text.

INPUT NOTE:
<note>
{note_text}
</note>
"""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "performance_status",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "ecog": {"type": "integer"},
                "ecog_source": {"type": "array", "items": {"type": "string"}},
                "ecog_confidence": {"type": "number"},
                "kps": {"type": "integer"},
                "kps_source": {"type": "array", "items": {"type": "string"}},
                "kps_confidence": {"type": "number"},
                "lansky": {"type": "integer"},
                "lansky_source": {"type": "array", "items": {"type": "string"}},
                "lansky_confidence": {"type": "number"},
            },
            "required": [
                "ecog", "ecog_source", "ecog_confidence",
                "kps", "kps_source", "kps_confidence",
                "lansky", "lansky_source", "lansky_confidence",
            ],
            "additionalProperties": False,
        },
    },
}

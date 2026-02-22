from pydantic import BaseModel
from typing import List
import json
import re


class DiagnosisItem(BaseModel):
    condition: str
    likelihood: str
    reason: str


class DifferentialDiagnosis(BaseModel):
    primary_suspected_condition: str
    differential_diagnosis: List[DiagnosisItem]
    recommended_tests: List[str]


def build_diagnosis_prompt(retrieved_chunks, user_query):
    context = "\n".join(retrieved_chunks)

    return f"""
You are a clinical diagnostic AI assistant.

Based ONLY on the provided medical context and symptoms,
generate a structured differential diagnosis.

Context:
{context}

User Symptoms:
{user_query}

Return STRICT JSON format:

{{
  "primary_suspected_condition": "...",
  "differential_diagnosis": [
    {{
      "condition": "...",
      "likelihood": "High/Moderate/Low",
      "reason": "..."
    }}
  ],
  "recommended_tests": ["..."]
}}
"""


def parse_diagnosis_output(raw_response: str):
    try:
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if not json_match:
            return None

        json_str = json_match.group(0)
        data = json.loads(json_str)

        return DifferentialDiagnosis(**data)

    except Exception:
        return None
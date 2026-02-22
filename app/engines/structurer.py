import json
from pydantic import ValidationError
from models.schemas import MedicalReport


def build_structuring_prompt(context_chunks):

    context = "\n\n".join(context_chunks)

    return f"""
You are a medical information extraction system.

Extract structured medical data from the document context below.

Return ONLY valid JSON in this format:

{{
  "patient_name": string or null,
  "age": integer or null,
  "gender": string or null,
  "diagnosis": string or null,
  "lab_values": [
      {{
          "test_name": string,
          "value": number,
          "unit": string or null,
          "reference_range": string or null,
          "is_abnormal": true/false/null
      }}
  ],
  "abnormal_markers": [string],
  "summary": string
}}

Document Context:
----------------
{context}
----------------
"""


def parse_structured_output(raw_output):

    try:
        # Extract JSON block
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        json_str = raw_output[start:end]

        data = json.loads(json_str)

        report = MedicalReport(**data)
        return report

    except (json.JSONDecodeError, ValidationError) as e:
        return None
from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class LabValue(BaseModel):
    test_name: str
    value: float
    unit: Optional[str]
    reference_range: Optional[str]
    is_abnormal: Optional[bool]


class MedicalReport(BaseModel):
    patient_name: Optional[str]
    age: Optional[int]
    gender: Optional[str]

    diagnosis: Optional[str]

    lab_values: List[LabValue] = Field(default_factory=list)
    abnormal_markers: List[str] = Field(default_factory=list)

    summary: Optional[str]
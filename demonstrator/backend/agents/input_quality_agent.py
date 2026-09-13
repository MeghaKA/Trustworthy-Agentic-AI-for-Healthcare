"""
Input / Data Quality Agent.

This is a faithful reproduction of NB9 Cell 14 ("AUTOMATED INPUT / DATA
QUALITY AGENT" -> `validate_input_record`), which is the authoritative,
general-purpose, live validator in the validated pipeline (as opposed to
NB8 Cell 15, which only re-derives static evidence about one fixed NB7
prototype patient and explicitly does not perform live 214-feature schema
validation). The completeness formulas, quality_status thresholds, and
safety_gate values below were verified against NB9's own executed cell
output before being reproduced here.

This agent does NOT:
  - fit, refit, or modify the locked model or preprocessor
  - fabricate, invent, or silently impute any missing patient value
  - make a diagnosis or treatment recommendation
  - override any downstream safety control

It only validates the structure and completeness of a supplied patient
record against the locked 214-feature schema and computes the same
safety-relevant flags NB9 computes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from backend.schema import LockedSchema, IDENTIFIER_COLUMN


@dataclass(frozen=True)
class InputQualityResult:
    """
    Structured result of the Input/Data Quality Agent, mirroring the exact
    JSON contract shape produced by NB9 Cell 14.
    """

    as_dict: dict[str, Any] = field(repr=False)

    @property
    def completeness(self) -> float:
        return self.as_dict["data_quality"]["completeness"]

    @property
    def quality_status(self) -> str:
        return self.as_dict["data_quality"]["quality_status"]

    @property
    def clinical_measurement_completeness(self) -> float:
        return self.as_dict["clinical_measurements"]["completeness"]

    @property
    def prediction_input_valid(self) -> bool:
        return self.as_dict["decision"]["prediction_input_valid"]

    @property
    def clinical_interpretation_allowed(self) -> bool:
        return self.as_dict["decision"]["clinical_interpretation_allowed"]

    @property
    def safety_gate(self) -> str:
        return self.as_dict["decision"]["safety_gate"]


def run_input_quality_agent(
    patient_record: dict[str, Any],
    schema: LockedSchema,
) -> InputQualityResult:
    """
    Validate one patient record against the locked 214-feature schema.

    patient_record: a dict that should contain a key for every feature in
    schema.locked_features (value None/NaN where not provided), and
    optionally an IDENTIFIER_COLUMN ("SEQN") key for provenance only. Any
    other key is treated as an unexpected field and blocks the input.

    This function performs no model inference of any kind.
    """
    record = dict(patient_record)

    locked_features = list(schema.locked_features)
    locked_feature_set = set(locked_features)
    clinical_measurements = list(schema.clinical_measurements)

    # -------------------------------------------------------------------
    # Field structure
    # -------------------------------------------------------------------
    supplied_fields = set(record.keys())

    unexpected_fields = sorted(
        supplied_fields - locked_feature_set - {IDENTIFIER_COLUMN}
    )
    missing_locked_fields = sorted(locked_feature_set - supplied_fields)

    # -------------------------------------------------------------------
    # Missingness (value-level, via pandas NA semantics)
    # -------------------------------------------------------------------
    missing_features: list[str] = []
    observed_features: list[str] = []

    for feature in locked_features:
        value = record.get(feature, np.nan)
        if pd.isna(value):
            missing_features.append(feature)
        else:
            observed_features.append(feature)

    total_features = len(locked_features)
    observed_count = len(observed_features)
    missing_count = len(missing_features)

    completeness = observed_count / total_features if total_features > 0 else 0.0

    # -------------------------------------------------------------------
    # Clinical measurement completeness
    # -------------------------------------------------------------------
    clinical_missing: list[str] = []
    clinical_observed: list[str] = []

    for feature in clinical_measurements:
        value = record.get(feature, np.nan)
        if pd.isna(value):
            clinical_missing.append(feature)
        else:
            clinical_observed.append(feature)

    clinical_total = len(clinical_measurements)
    clinical_observed_count = len(clinical_observed)
    clinical_missing_count = len(clinical_missing)

    clinical_completeness = (
        clinical_observed_count / clinical_total if clinical_total > 0 else 0.0
    )

    # -------------------------------------------------------------------
    # Numeric validity
    # -------------------------------------------------------------------
    invalid_numeric_fields: list[str] = []

    for feature in schema.numeric_features:
        value = record.get(feature, np.nan)
        if pd.isna(value):
            continue
        try:
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                invalid_numeric_fields.append(feature)
        except (TypeError, ValueError):
            invalid_numeric_fields.append(feature)

    # -------------------------------------------------------------------
    # Quality status (exact NB9 thresholds)
    # -------------------------------------------------------------------
    if completeness < 0.50 or clinical_observed_count == 0:
        quality_status = "VERY_LOW"
    elif completeness < 0.75 or clinical_completeness < 1.0:
        quality_status = "LOW"
    elif completeness < 0.90:
        quality_status = "MODERATE"
    else:
        quality_status = "HIGH"

    # -------------------------------------------------------------------
    # Safety decision (exact NB9 rule order)
    # -------------------------------------------------------------------
    if len(unexpected_fields) > 0 or len(invalid_numeric_fields) > 0:
        prediction_input_valid = False
        clinical_interpretation_allowed = False
        safety_gate = "INVALID_INPUT_SCHEMA"
    elif clinical_observed_count == 0:
        prediction_input_valid = True
        clinical_interpretation_allowed = False
        safety_gate = "INSUFFICIENT_CLINICAL_MEASUREMENTS"
    elif clinical_observed_count < clinical_total:
        prediction_input_valid = True
        clinical_interpretation_allowed = False
        safety_gate = "PARTIAL_CLINICAL_MEASUREMENTS"
    else:
        prediction_input_valid = True
        clinical_interpretation_allowed = True
        safety_gate = "INPUT_VALID_FOR_NEXT_STAGE"

    result_dict = {
        "agent": "Input_Data_Quality_Agent",
        "agent_version": "NB9_INPUT_AGENT_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "input_schema": {
            "locked_features": total_features,
            "supplied_fields": len(supplied_fields),
            "unexpected_fields": unexpected_fields,
            "missing_locked_fields_count": len(missing_locked_fields),
            "missing_locked_fields": missing_locked_fields,
        },
        "data_quality": {
            "observed_features": observed_count,
            "missing_features": missing_count,
            "completeness": completeness,
            "quality_status": quality_status,
            "invalid_numeric_fields": invalid_numeric_fields,
        },
        "clinical_measurements": {
            "total": clinical_total,
            "observed": clinical_observed,
            "missing": clinical_missing,
            "observed_count": clinical_observed_count,
            "missing_count": clinical_missing_count,
            "completeness": clinical_completeness,
        },
        "decision": {
            "prediction_input_valid": prediction_input_valid,
            "clinical_interpretation_allowed": clinical_interpretation_allowed,
            "safety_gate": safety_gate,
        },
        "governance": {
            "missing_values_fabricated": False,
            "patient_values_imputed": False,
            "clinical_measurements_invented": False,
            "diagnosis_performed": False,
            "treatment_recommended": False,
            "safety_override_performed": False,
        },
    }

    return InputQualityResult(as_dict=result_dict)

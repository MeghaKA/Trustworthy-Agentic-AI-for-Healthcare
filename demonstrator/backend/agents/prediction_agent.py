"""
Prediction Agent.

This is a faithful reproduction of NB9 Cell 15 ("AUTOMATED PREDICTION
AGENT"), adapted to consume this application's already-loaded Governance
and ModelBundle objects instead of re-loading artifacts from disk, and to
run on a real, dynamically-submitted patient record instead of a fixed
demonstration row.

This agent does NOT:
  - fit, refit, or modify the locked model or preprocessor
    (preprocessor.transform is used; .fit()/.fit_transform() are never
    called anywhere in this module)
  - modify the locked decision threshold (read once from
    ModelBundle.locked_threshold, itself sourced from governance/NB3
    metadata — no second source of truth is introduced here)
  - fabricate, invent, or silently impute any missing patient value
    (missing fields are passed through as NaN; nothing is filled in)
  - make a diagnosis or treatment recommendation
  - override the Input/Data Quality Agent's safety flags

Safety-gate handling (this is a governance decision made in this phase,
not something NB9's fixed demo needed to handle):
  - If the Input/Data Quality Agent reports
    decision.prediction_input_valid == False (safety_gate ==
    INVALID_INPUT_SCHEMA), this agent does NOT call
    preprocessor.transform() or model.predict_proba() at all. No
    computation is attempted on a structurally invalid input.
  - If prediction_input_valid == True but clinical_interpretation_allowed
    == False (missing/partial clinical measurements), this agent DOES
    compute a prediction — exactly as NB9 does — but the result carries
    interpretation_status = CLINICAL_INTERPRETATION_BLOCKED and the
    corresponding handoff_status, so downstream/UI code can refuse to
    present it as a clinical interpretation.

The model output is a technical model output only. It is not a
calibrated individual clinical risk estimate, not a diagnosis, and not
medical advice, regardless of clinical_interpretation_allowed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from backend.governance import Governance
from backend.model_loader import ModelBundle
from backend.schema import LockedSchema, IDENTIFIER_COLUMN
from backend.agents.input_quality_agent import InputQualityResult


@dataclass(frozen=True)
class PredictionResult:
    """Structured result of the Prediction Agent, mirroring the NB9 Cell 15 JSON contract."""

    as_dict: dict[str, Any] = field(repr=False)

    @property
    def prediction_computed(self) -> bool:
        return self.as_dict["prediction"]["prediction_computed"]

    @property
    def predicted_probability(self) -> float | None:
        return self.as_dict["prediction"]["predicted_probability"]

    @property
    def predicted_class(self) -> int | None:
        return self.as_dict["prediction"]["predicted_class"]

    @property
    def prediction_label(self) -> str | None:
        return self.as_dict["prediction"]["prediction_label"]

    @property
    def clinical_interpretation_allowed(self) -> bool:
        return self.as_dict["safety_inheritance"]["clinical_interpretation_allowed"]

    @property
    def interpretation_status(self) -> str:
        return self.as_dict["safety_inheritance"]["interpretation_status"]


def _build_locked_input_frame(
    patient_record: dict[str, Any],
    schema: LockedSchema,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame with exactly the locked 214 columns, in
    the locked order, values None -> NaN. Never fills a value that was not
    already present in patient_record.
    """
    row = {
        feature: (np.nan if patient_record.get(feature) is None else patient_record.get(feature))
        for feature in schema.locked_features
    }
    return pd.DataFrame([row], columns=list(schema.locked_features))


def run_prediction_agent(
    patient_record: dict[str, Any],
    schema: LockedSchema,
    model_bundle: ModelBundle,
    governance: Governance,
    input_quality_result: InputQualityResult,
) -> PredictionResult:
    """
    Run the Prediction Agent on one patient record.

    model_bundle must already have is_safe_to_use == True (the caller is
    expected to have blocked the entire application already if not — this
    function does not re-check artifact integrity).
    """
    decision = input_quality_result.as_dict["decision"]
    prediction_input_valid: bool = decision["prediction_input_valid"]
    clinical_interpretation_allowed: bool = decision["clinical_interpretation_allowed"]

    clinical_measurements = input_quality_result.as_dict["clinical_measurements"]
    clinical_observed_count = clinical_measurements["observed_count"]
    clinical_missing_count = clinical_measurements["missing_count"]
    input_quality_status = input_quality_result.as_dict["data_quality"]["quality_status"]

    seqn_value = patient_record.get(IDENTIFIER_COLUMN)

    model_provenance = {
        "model_path": str(governance.model_artifact.path),
        "preprocessor_path": str(governance.preprocessor_artifact.path),
        "metadata_path": str(governance.metadata_path),
        "model_class": type(model_bundle.model).__name__,
        "locked_threshold": model_bundle.locked_threshold,
        "threshold_selection_method": governance.threshold_selection_method,
        "test_set_used_for_threshold_selection": governance.raw_metadata["threshold_information"][
            "test_set_used_for_threshold_selection"
        ],
    }

    metadata_sha256 = hashlib.sha256(governance.metadata_path.read_bytes()).hexdigest()
    artifact_hashes = {
        "model_sha256": model_bundle.model_integrity.actual_sha256,
        "preprocessor_sha256": model_bundle.preprocessor_integrity.actual_sha256,
        "metadata_sha256": metadata_sha256,
    }

    base_governance_block = {
        "model_retrained": False,
        "preprocessor_refit": False,
        "threshold_modified": False,
        "missing_values_fabricated": False,
        "patient_values_manually_imputed": False,
        "clinical_measurements_invented": False,
        "diagnosis_performed": False,
        "treatment_recommended": False,
        "safety_override_performed": False,
    }

    if not prediction_input_valid:
        # Hard safety gate: do NOT call transform() or predict_proba() on a
        # structurally invalid input. No computation is attempted.
        result_dict = {
            "agent": "Prediction_Agent",
            "agent_version": "PHASE3_PREDICTION_AGENT_V1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "input_provenance": {
                "SEQN": seqn_value,
                "source": "Structured patient input form (Phase 2)",
                "clinical_measurements_intentionally_missing": clinical_observed_count == 0,
            },
            "model_provenance": model_provenance,
            "input_schema": {
                "raw_features": len(schema.locked_features),
                "processed_features": None,
                "schema_matches_locked_preprocessor": False,
            },
            "prediction": {
                "prediction_computed": False,
                "predicted_probability": None,
                "threshold": model_bundle.locked_threshold,
                "predicted_class": None,
                "prediction_label": None,
            },
            "safety_inheritance": {
                "input_quality_status": input_quality_status,
                "clinical_measurements_observed": clinical_observed_count,
                "clinical_measurements_missing": clinical_missing_count,
                "clinical_interpretation_allowed": clinical_interpretation_allowed,
                "interpretation_status": "PREDICTION_NOT_COMPUTED_INVALID_SCHEMA",
                "handoff_status": "BLOCKED_BEFORE_PREPROCESSING_INVALID_INPUT_SCHEMA",
            },
            "governance": base_governance_block,
            "artifact_hashes": artifact_hashes,
        }
        return PredictionResult(as_dict=result_dict)

    # -------------------------------------------------------------------
    # Build exact model input (locked column order, no fabricated values)
    # -------------------------------------------------------------------
    X_input = _build_locked_input_frame(patient_record, schema)

    if list(X_input.columns) != list(schema.locked_features):
        raise ValueError("Prediction input schema does not match locked NB3 schema.")

    # -------------------------------------------------------------------
    # Transform using the already-fitted, locked preprocessor. NO FIT.
    # -------------------------------------------------------------------
    X_transformed = model_bundle.preprocessor.transform(X_input)

    processed_feature_count = int(X_transformed.shape[1])

    # -------------------------------------------------------------------
    # Locked model prediction
    # -------------------------------------------------------------------
    predicted_probability = float(model_bundle.model.predict_proba(X_transformed)[0, 1])
    predicted_class = int(predicted_probability >= model_bundle.locked_threshold)
    prediction_label = "MODEL_CLASS_1" if predicted_class == 1 else "MODEL_CLASS_0"

    if clinical_interpretation_allowed:
        interpretation_status = "CLINICAL_INTERPRETATION_MAY_PROCEED"
        handoff_status = "PREDICTION_READY_FOR_NEXT_GOVERNED_AGENT"
    else:
        interpretation_status = "CLINICAL_INTERPRETATION_BLOCKED"
        handoff_status = "PREDICTION_COMPUTATION_ALLOWED_BUT_CLINICAL_INTERPRETATION_BLOCKED"

    result_dict = {
        "agent": "Prediction_Agent",
        "agent_version": "PHASE3_PREDICTION_AGENT_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "input_provenance": {
            "SEQN": seqn_value,
            "source": "Structured patient input form (Phase 2)",
            "clinical_measurements_intentionally_missing": clinical_observed_count == 0,
        },
        "model_provenance": model_provenance,
        "input_schema": {
            "raw_features": len(schema.locked_features),
            "processed_features": processed_feature_count,
            "schema_matches_locked_preprocessor": True,
        },
        "prediction": {
            "prediction_computed": True,
            "predicted_probability": predicted_probability,
            "threshold": model_bundle.locked_threshold,
            "predicted_class": predicted_class,
            "prediction_label": prediction_label,
        },
        "safety_inheritance": {
            "input_quality_status": input_quality_status,
            "clinical_measurements_observed": clinical_observed_count,
            "clinical_measurements_missing": clinical_missing_count,
            "clinical_interpretation_allowed": clinical_interpretation_allowed,
            "interpretation_status": interpretation_status,
            "handoff_status": handoff_status,
        },
        "governance": base_governance_block,
        "artifact_hashes": artifact_hashes,
    }

    return PredictionResult(as_dict=result_dict)

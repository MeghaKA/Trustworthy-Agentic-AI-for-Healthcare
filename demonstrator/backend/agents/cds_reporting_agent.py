"""
CDS/Reporting Agent.

Phase 7: a pure reporting/presentation layer that assembles the already-
governed outputs of Phases 2-6 (Input/Data Quality, Prediction,
Explainability, Trust/Fairness, Safety) into one structured, deterministic
"decision_support_report" object.

THIS AGENT DOES NOT:
  - call .transform(), .predict(), .predict_proba(), .fit(), or
    .fit_transform() — it never touches the locked model or preprocessor
    objects, directly or indirectly.
  - recompute any value already produced by an upstream agent. Every
    number in this report is read verbatim from an upstream result dict
    (input_quality_result, prediction_result, explainability_result,
    trust_fairness_result, safety_result) or from governance/schema
    constants that already existed before this phase.
  - independently decide clinical_interpretation_allowed, or any other
    *_allowed flag. It reads safety_result's flags and never overrides
    them. If safety_result says a flag is False, this agent's report
    reflects False — there is no code path in this module that can flip
    it to True.
  - build a second prediction or preprocessing pipeline of any kind.
  - diagnose, recommend treatment or medication, or claim medical
    certainty, clinical validation, or that the model replaces a
    clinician — anywhere, even when clinical_interpretation_allowed is
    True.
  - convert population-level Trust/Fairness evidence into a claim about
    this individual patient.

GATING (mirrors Safety Agent's authoritative flags exactly):
  - technical_prediction_allowed == False -> Section 3 (Technical Model
    Output) contains no probability/classification data at all, only a
    blocked-state message.
  - technical_explanation_allowed == False -> Section 4 (Model
    Explanation) contains no contribution data at all, only a
    blocked-state message.
  - clinical_interpretation_allowed == False -> Section 7 (Decision-
    Support Interpretation) contains the required restriction statement
    and the relevant reason codes, and nothing resembling a clinical
    interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.governance import Governance
from backend.schema import LockedSchema
from backend.agents.input_quality_agent import InputQualityResult
from backend.agents.prediction_agent import PredictionResult
from backend.agents.explainability_agent import ExplainabilityResult
from backend.agents.trust_fairness_agent import TrustFairnessResult
from backend.agents.safety_agent import SafetyResult

CLINICAL_INTERPRETATION_RESTRICTED_STATEMENT = (
    "Decision-support interpretation is not available under the current "
    "demonstrator safety gate."
)

CLINICAL_INTERPRETATION_PERMITTED_STATEMENT = (
    "The locked research model produced this technical classification "
    "under the configured threshold. This output is intended for "
    "research demonstration and decision-support workflow evaluation "
    "and should not be interpreted as a medical diagnosis or treatment "
    "recommendation."
)

# Additional Trust/Fairness boundary this report adds on top of the
# Safety Agent's own TRUST_FAIRNESS_BOUNDARIES list (reused verbatim from
# safety_result). This does not modify safety_agent.py; it is this
# reporting layer's own supplementary limitation statement.
HISTORICAL_EVIDENCE_BOUNDARY = (
    "Historical population-level evaluation evidence (NB5/NB6) describes "
    "the locked model's past held-out test cohort. It does not guarantee "
    "the validity of the current case."
)

LIMITATIONS_TEMPLATE_STATIC = [
    "This is a research demonstrator, not a certified, approved, or "
    "regulated clinical software product.",
    "The locked model was trained and evaluated using NHANES-derived "
    "data; it has not been evaluated on any other population or clinical "
    "setting by this demonstrator.",
    "Population-level evaluation evidence describes the locked model's "
    "held-out NHANES test cohort only, not this individual patient.",
    "The model's probability output is not necessarily a calibrated "
    "clinical probability of disease.",
    "Population-level fairness evidence does not establish individual-"
    "level fairness for this patient.",
    "Explanation consistency evidence (NB6) is a trustworthiness signal "
    "only and does not establish clinical correctness.",
    "No diagnosis or treatment recommendation is made anywhere in this "
    "report.",
    "External clinical validation of this model has not been established "
    "by this demonstrator.",
    HISTORICAL_EVIDENCE_BOUNDARY,
]


@dataclass(frozen=True)
class CDSReportingResult:
    """Structured result of the CDS/Reporting Agent — the decision_support_report."""

    as_dict: dict[str, Any] = field(repr=False)

    @property
    def report(self) -> dict[str, Any]:
        return self.as_dict["decision_support_report"]

    @property
    def clinical_interpretation_provided(self) -> bool:
        return self.report["decision_support_interpretation"]["clinical_interpretation_provided"]


def _build_assessment_overview(
    governance: Governance,
    prediction_result: PredictionResult,
    explainability_result: ExplainabilityResult,
    safety_result: SafetyResult,
) -> dict[str, Any]:
    model_provenance = prediction_result.as_dict["model_provenance"]
    return {
        "session_identifier": prediction_result.as_dict["input_provenance"].get("SEQN"),
        "model_class": model_provenance["model_class"],
        "model_artifact_path": model_provenance["model_path"],
        "preprocessor_artifact_path": model_provenance["preprocessor_path"],
        "threshold": governance.locked_threshold,
        "prediction_available": prediction_result.prediction_computed,
        "explanation_available": explainability_result.explanation_computed,
        "clinical_interpretation_available": safety_result.clinical_interpretation_allowed,
        "safety_decision": safety_result.safety_decision,
    }


def _build_input_quality_section(input_quality_result: InputQualityResult) -> dict[str, Any]:
    r = input_quality_result.as_dict
    return {
        "overall_model_input_completeness": r["data_quality"]["completeness"],
        "quality_status": r["data_quality"]["quality_status"],
        "observed_feature_count": r["data_quality"]["observed_features"],
        "missing_feature_count": r["data_quality"]["missing_features"],
        "clinical_measurement_completeness": r["clinical_measurements"]["completeness"],
        "clinical_measurements_observed_count": r["clinical_measurements"]["observed_count"],
        "clinical_measurements_missing": r["clinical_measurements"]["missing"],
        "schema_valid": r["decision"]["prediction_input_valid"],
        "input_quality_safety_gate": r["decision"]["safety_gate"],
    }


def _build_technical_model_output_section(
    prediction_result: PredictionResult,
    safety_result: SafetyResult,
) -> dict[str, Any]:
    if not safety_result.technical_prediction_allowed:
        return {
            "technical_prediction_allowed": False,
            "message": (
                "Technical model output is not displayed because the "
                "Safety Agent has not permitted it for this input."
            ),
        }

    pred = prediction_result.as_dict["prediction"]
    return {
        "technical_prediction_allowed": True,
        "model_estimated_probability": pred["predicted_probability"],
        "threshold": pred["threshold"],
        "technical_model_classification": pred["prediction_label"],
        "disclaimer": (
            "Technical model output only. Not a diagnosis. Not a "
            "clinically validated probability of disease."
        ),
    }


def _build_explanation_section(
    explainability_result: ExplainabilityResult,
    safety_result: SafetyResult,
) -> dict[str, Any]:
    if not safety_result.technical_explanation_allowed:
        return {
            "technical_explanation_allowed": False,
            "message": (
                "Model explanation is not displayed because the Safety "
                "Agent has not permitted it for this input."
            ),
        }

    e = explainability_result.as_dict["explanation"]
    audit_summary = safety_result.as_dict["input_observation_summary"]

    def _as_model_contributions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "feature_label": row["feature_label"],
                "model_contribution": row["contribution"],
                "direction": row["direction"],
                "contribution_type": "MODEL_CONTRIBUTION",
            }
            for row in rows
        ]

    return {
        "technical_explanation_allowed": True,
        "explanation_method": explainability_result.as_dict["explanation_method"],
        "technical_disclaimer": explainability_result.as_dict["technical_disclaimer"],
        "intercept": e["intercept"],
        "total_logit": e["total_logit"],
        "reconstructed_logit": e["reconstructed_logit"],
        "reconstruction_error": e["reconstruction_error"],
        "reconstruction_within_tolerance": e["reconstruction_within_tolerance"],
        "top_positive_model_contributions": _as_model_contributions(
            explainability_result.as_dict["contributions"]["top_positive"]
        ),
        "top_negative_model_contributions": _as_model_contributions(
            explainability_result.as_dict["contributions"]["top_negative"]
        ),
        "contribution_audit_scope": audit_summary["contribution_audit_scope"],
        "contribution_audit_limit": audit_summary["contribution_audit_limit"],
        "total_transformed_features": audit_summary["total_transformed_features"],
        "observed_missing_imputed_note": audit_summary["contribution_audit_note"],
        "note_on_language": (
            "Each row above is a model contribution (coefficient × "
            "transformed feature value), also describable as 'feature "
            "contribution to the model output'. It is never a clinical "
            "cause, disease cause, or patient risk factor conclusion."
        ),
    }


def _build_trust_fairness_section(
    trust_fairness_result: TrustFairnessResult,
    safety_result: SafetyResult,
) -> dict[str, Any]:
    pop_status = safety_result.as_dict["population_evidence_status"]
    tf = trust_fairness_result.as_dict
    return {
        "current_session_status": pop_status["current_session_status"],
        "static_reference_status": pop_status["static_reference_status"],
        "population_data_available": pop_status["population_data_available"],
        "static_population_reference": tf["evaluation_scope"]["static_population_reference"],
        "boundaries": list(pop_status["boundaries"]) + [HISTORICAL_EVIDENCE_BOUNDARY],
    }


def _build_safety_governance_section(safety_result: SafetyResult) -> dict[str, Any]:
    s = safety_result.as_dict
    return {
        "safety_decision": s["safety_decision"],
        "clinical_interpretation_allowed": s["clinical_interpretation_allowed"],
        "technical_prediction_allowed": s["technical_prediction_allowed"],
        "technical_explanation_allowed": s["technical_explanation_allowed"],
        "reason_codes": s["reason_codes"],
        "safety_flags": s["safety_flags"],
        "observed_missing_summary": {
            "observed_count": s["input_observation_summary"]["observed_count"],
            "missing_count": s["input_observation_summary"]["missing_count"],
            "total_raw_features": s["input_observation_summary"]["total_raw_features"],
        },
        "population_evidence_status": s["population_evidence_status"],
        "audit_summary": s["audit_summary"],
    }


def _build_decision_support_interpretation_section(safety_result: SafetyResult) -> dict[str, Any]:
    if not safety_result.clinical_interpretation_allowed:
        return {
            "clinical_interpretation_provided": False,
            "statement": CLINICAL_INTERPRETATION_RESTRICTED_STATEMENT,
            "reason_codes": safety_result.as_dict["reason_codes"],
        }

    return {
        "clinical_interpretation_provided": True,
        "statement": CLINICAL_INTERPRETATION_PERMITTED_STATEMENT,
        "clinical_gate_disclaimer": safety_result.as_dict["clinical_gate_disclaimer"],
        "explicit_non_claims": {
            "diagnosis_provided": False,
            "treatment_recommended": False,
            "medication_recommended": False,
            "medical_certainty_claimed": False,
            "clinical_validation_claimed": False,
            "replaces_clinician": False,
        },
    }


def _build_limitations_section(
    input_quality_result: InputQualityResult,
    safety_result: SafetyResult,
) -> list[str]:
    limitations = list(LIMITATIONS_TEMPLATE_STATIC)
    dq = input_quality_result.as_dict["data_quality"]
    cm = input_quality_result.as_dict["clinical_measurements"]
    limitations.append(
        f"Current input completeness for this session: "
        f"{dq['completeness'] * 100:.2f}% of 214 model features, "
        f"{cm['completeness'] * 100:.2f}% of 7 clinical measurements."
    )
    note = safety_result.as_dict["input_observation_summary"]["contribution_audit_note"]
    if note:
        limitations.append(note)
    return limitations


def _build_audit_section(
    governance: Governance,
    schema: LockedSchema,
    prediction_result: PredictionResult,
    safety_result: SafetyResult,
) -> dict[str, Any]:
    hashes = prediction_result.as_dict["artifact_hashes"]
    return {
        "model_sha256": hashes["model_sha256"],
        "preprocessor_sha256": hashes["preprocessor_sha256"],
        "metadata_sha256": hashes["metadata_sha256"],
        "threshold": governance.locked_threshold,
        "raw_feature_count": len(schema.locked_features),
        "transformed_feature_count": governance.processed_feature_count,
        "safety_decision": safety_result.safety_decision,
        "report_generation_status": "GENERATED",
    }


def run_cds_reporting_agent(
    governance: Governance,
    schema: LockedSchema,
    input_quality_result: InputQualityResult,
    prediction_result: PredictionResult,
    explainability_result: ExplainabilityResult,
    trust_fairness_result: TrustFairnessResult,
    safety_result: SafetyResult,
) -> CDSReportingResult:
    """
    Assemble the decision_support_report. Always succeeds and always
    returns a structured report — sections are populated or replaced
    with an explicit blocked-state message depending on the Safety
    Agent's flags, never omitted silently and never bypassed.
    """
    report = {
        "assessment_overview": _build_assessment_overview(
            governance, prediction_result, explainability_result, safety_result
        ),
        "input_quality": _build_input_quality_section(input_quality_result),
        "technical_model_output": _build_technical_model_output_section(
            prediction_result, safety_result
        ),
        "model_explanation": _build_explanation_section(explainability_result, safety_result),
        "trust_fairness_evidence": _build_trust_fairness_section(
            trust_fairness_result, safety_result
        ),
        "safety_governance": _build_safety_governance_section(safety_result),
        "decision_support_interpretation": _build_decision_support_interpretation_section(
            safety_result
        ),
        "limitations": _build_limitations_section(input_quality_result, safety_result),
        "audit_information": _build_audit_section(
            governance, schema, prediction_result, safety_result
        ),
    }

    result_dict = {
        "agent_name": "CDS_Reporting_Agent",
        "phase": "Phase 7",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "decision_support_report": report,
    }

    return CDSReportingResult(as_dict=result_dict)

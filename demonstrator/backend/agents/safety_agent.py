"""
Safety Agent.

EXECUTION ORDER (documented explicitly per governance requirement):

    Input Quality Agent
        -> Prediction Agent
        -> Explainability Agent
        -> Trust/Fairness Agent
        -> Safety Agent
        -> Display

Two distinct gates exist in this pipeline, and they are not the same
thing:
  - The Prediction Agent (Phase 3) already HARD-BLOCKS model inference
    at its own level: if the Input/Data Quality Agent reports
    safety_gate == INVALID_INPUT_SCHEMA, the Prediction Agent refuses to
    call preprocessor.transform() or model.predict_proba() at all
    (verified by spy-based tests). This is a compute-time gate.
  - The Safety Agent (this module, Phase 6) is the FINAL, AUTHORITATIVE
    DOWNSTREAM DISPLAY GATE. It runs after all four upstream agents have
    already executed (or refused to execute) and decides what the UI is
    allowed to show. It does not re-run, undo, or duplicate the
    Prediction Agent's compute-time block — it independently confirms
    and enforces the same conclusion at the display layer, and the UI
    (app.py) consults this agent's flags, not just each upstream agent's
    own flags, before rendering anything. This is what makes the Safety
    Agent's decision impossible to bypass through the normal UI
    workflow, even though by construction its flags will normally agree
    with the upstream agents' own flags.

This is the mandatory governance control point that determines what may
be shown as (1) technical model output, (2) technical explanation,
(3) population-level trust/fairness evidence, and (4) clinical
interpretation.

This agent is a pure governance layer. It performs NO model computation
of any kind: no .transform(), no .predict(), no .predict_proba(), no
.fit(), no .fit_transform(). It does not touch the locked model or
preprocessor objects at all. It only reads the already-computed
structured results from the four upstream agents (and the raw patient
record, for the observed/missing audit in section 5 below) and applies a
deterministic set of rules to them.

CORE GOVERNANCE INVARIANT — SAFETY MAY ONLY TIGHTEN, NEVER LOOSEN:
Every "*_allowed" flag this agent produces is computed as the logical AND
of (a) the corresponding upstream flag(s) and (b) this agent's own rule
for the current clinical-measurement completeness state. This means the
Safety Agent can never grant a permission an upstream agent withheld,
even if this agent's own rule alone would have allowed it. It can only
agree with upstream or restrict further.

RULE HIERARCHY (highest priority first):
  1. Input/Data Quality Agent reports safety_gate == INVALID_INPUT_SCHEMA
     -> safety_decision = "BLOCK": no technical prediction, no technical
     explanation, no clinical interpretation. The model must not be
     called, and indeed was not (the Prediction Agent already refused to
     run in this case; this agent verifies and reports that, it does not
     re-decide it).
  2. clinical_observed_count == 0
     -> safety_decision = "TECHNICAL_ONLY": technical prediction and
     technical explanation may be shown (if upstream agents actually
     computed them), but clinical interpretation is blocked, with an
     explicit reason naming that zero of the seven required clinical
     measurements were observed.
  3. 0 < clinical_observed_count < 7
     -> safety_decision = "TECHNICAL_ONLY": same as above, with an
     explicit reason naming which of the seven clinical measurements are
     missing.
  4. clinical_observed_count == 7 (and schema valid, and upstream agrees)
     -> safety_decision = "CLINICAL_INTERPRETATION_GATE_PASSED". This
     phrase means ONLY that this demonstrator's predefined input-
     completeness gate has passed. It does NOT mean medically validated,
     clinically validated, diagnostic, or safe for clinical deployment.

OBSERVED / MISSING / IMPUTED:
The locked preprocessor contains fitted imputation behavior (e.g. a
fitted SimpleImputer), so the Explainability Agent can produce a
non-zero contribution for a transformed feature even when the
corresponding raw patient input was never supplied. This agent builds an
explicit audit distinguishing:
  - OBSERVED: the raw feature had a non-missing value in the submitted
    patient record.
  - MISSING: the raw feature was not supplied (None/NaN) in the
    submitted patient record.
  - IMPUTED (at the transformed-feature / contribution level only): a
    contribution whose underlying raw feature was MISSING, meaning the
    value used in that contribution came from the locked preprocessor's
    already-fitted imputation strategy, not from the patient's own
    submitted data.
This agent does not add, change, or invent any imputation. It only reads
which raw fields were supplied and cross-references that against the
transformed feature names the Explainability Agent already computed.

IMPORTANT SCOPE LIMIT: the observed/imputed audit below is a
TOP-CONTRIBUTION AUDIT, not an exhaustive audit. It only covers the
CONTRIBUTION_AUDIT_LIMIT highest-ranked contributions the Explainability
Agent already surfaced (top positive + top negative), not all 517
transformed features. This is stated explicitly in the result's
contribution_audit_scope, contribution_audit_limit, and
contribution_audit_note fields so nothing implies full coverage.

This agent does not recompute population fairness metrics, does not
fabricate missing patient values, does not perform diagnosis, does not
provide treatment recommendations, and cannot override an upstream
safety failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from backend.schema import LockedSchema
from backend.agents.input_quality_agent import InputQualityResult
from backend.agents.prediction_agent import PredictionResult
from backend.agents.explainability_agent import ExplainabilityResult
from backend.agents.trust_fairness_agent import TrustFairnessResult

# How many of the top explainability contributions (already ranked by the
# Explainability Agent) to include in the observed/imputed contribution
# audit. This does not limit what the Explainability Agent computes —
# only how many rows this agent's audit table surfaces.
CONTRIBUTION_AUDIT_LIMIT = 30

TRUST_FAIRNESS_BOUNDARIES = [
    "Population-level fairness evidence is not an individual patient "
    "fairness determination.",
    "Explanation consistency evidence is not clinical correctness.",
    "Equal or similar metrics across groups are not proof of fairness.",
    "A single synthetic or real patient input is not population evidence, "
    "and population evidence is never generalized into a claim about "
    "this specific patient.",
]

EXPLANATION_INTERPRETATION_BOUNDARIES = [
    "A contribution is a model contribution: coefficient times "
    "transformed feature value. It is not a medical cause, not a "
    "clinical risk factor conclusion, not a causal relationship, not a "
    "diagnosis, and not treatment guidance.",
    "Use 'model contribution' or 'model-associated feature "
    "contribution' — never 'clinical risk factor' or 'cause of disease'.",
]


def _raw_feature_from_transformed_name(transformed_name: str, schema: LockedSchema) -> str:
    """
    Resolve a transformed feature name (e.g. 'numeric__RIDAGEYR' or
    "categorical__BPAOARM_b'L'") back to its raw NB3 feature name, using
    only the preprocessor's own naming convention and the known set of
    raw feature names in the locked schema. Does not invent a mapping —
    if no known raw feature matches, the transformed name is returned
    unchanged.
    """
    if transformed_name.startswith("numeric__"):
        candidate = transformed_name[len("numeric__"):]
        return candidate if candidate in schema.locked_features else transformed_name

    if transformed_name.startswith("categorical__"):
        remainder = transformed_name[len("categorical__"):]
        # Match against known categorical raw features, longest name first
        # to avoid a shorter name accidentally matching a prefix of a
        # longer one.
        for raw_col in sorted(schema.categorical_features, key=len, reverse=True):
            if remainder.startswith(raw_col + "_") or remainder == raw_col:
                return raw_col
        return remainder

    return transformed_name


def _build_raw_observation_status(
    patient_record: dict[str, Any],
    schema: LockedSchema,
) -> dict[str, str]:
    """
    OBSERVED / MISSING status for every one of the 214 locked raw
    features, based only on the patient record actually submitted. No
    model or preprocessor involvement.
    """
    status: dict[str, str] = {}
    for feature in schema.locked_features:
        value = patient_record.get(feature)
        status[feature] = "MISSING" if pd.isna(value) else "OBSERVED"
    return status


@dataclass(frozen=True)
class SafetyResult:
    """Structured result of the Safety Agent."""

    as_dict: dict[str, Any] = field(repr=False)

    @property
    def safety_decision(self) -> str:
        return self.as_dict["safety_decision"]

    @property
    def clinical_interpretation_allowed(self) -> bool:
        return self.as_dict["clinical_interpretation_allowed"]

    @property
    def technical_prediction_allowed(self) -> bool:
        return self.as_dict["technical_prediction_allowed"]

    @property
    def technical_explanation_allowed(self) -> bool:
        return self.as_dict["technical_explanation_allowed"]


def run_safety_agent(
    patient_record: dict[str, Any],
    schema: LockedSchema,
    input_quality_result: InputQualityResult,
    prediction_result: PredictionResult,
    explainability_result: ExplainabilityResult,
    trust_fairness_result: TrustFairnessResult,
) -> SafetyResult:
    """
    Run the Safety Agent. Requires all four upstream results (this agent
    is the last governance step before display, run after the full
    upstream chain has executed).
    """
    iq_decision = input_quality_result.as_dict["decision"]
    iq_clinical = input_quality_result.as_dict["clinical_measurements"]

    input_schema_valid = bool(iq_decision["prediction_input_valid"])
    clinical_observed_count = int(iq_clinical["observed_count"])
    clinical_total = int(iq_clinical["total"])
    clinical_missing_names = list(iq_clinical["missing"])

    pred_dict = prediction_result.as_dict["prediction"]
    pred_safety = prediction_result.as_dict["safety_inheritance"]
    expl_dict = explainability_result.as_dict["explanation"]

    # -------------------------------------------------------------------
    # Rule hierarchy (section 9). Each branch sets safety_decision and
    # reason_codes; the *_allowed flags are computed afterward, always as
    # an AND against upstream, so this if/elif chain determines the
    # local rule but never by itself grants a permission upstream denied.
    # -------------------------------------------------------------------
    if not input_schema_valid:
        safety_decision = "BLOCK"
        reason_codes = ["INVALID_INPUT_SCHEMA"]
        rule_allows_technical_prediction = False
        rule_allows_technical_explanation = False
        rule_allows_clinical_interpretation = False
    elif clinical_observed_count == 0:
        safety_decision = "TECHNICAL_ONLY"
        reason_codes = ["NO_CLINICAL_MEASUREMENTS_OBSERVED"]
        rule_allows_technical_prediction = True
        rule_allows_technical_explanation = True
        rule_allows_clinical_interpretation = False
    elif clinical_observed_count < clinical_total:
        safety_decision = "TECHNICAL_ONLY"
        reason_codes = [
            "PARTIAL_CLINICAL_MEASUREMENTS",
            "MISSING_CLINICAL_MEASUREMENTS: " + ", ".join(clinical_missing_names),
        ]
        rule_allows_technical_prediction = True
        rule_allows_technical_explanation = True
        rule_allows_clinical_interpretation = False
    else:
        safety_decision = "CLINICAL_INTERPRETATION_GATE_PASSED"
        reason_codes = ["ALL_SEVEN_CLINICAL_MEASUREMENTS_OBSERVED"]
        rule_allows_technical_prediction = True
        rule_allows_technical_explanation = True
        rule_allows_clinical_interpretation = True

    # -------------------------------------------------------------------
    # Tighten-only composition against upstream flags. This agent can
    # never end up with an *_allowed flag that is True when an upstream
    # agent's corresponding flag was False.
    # -------------------------------------------------------------------
    technical_prediction_allowed = rule_allows_technical_prediction and bool(
        pred_dict["prediction_computed"]
    )
    technical_explanation_allowed = rule_allows_technical_explanation and bool(
        expl_dict["explanation_computed"]
    )
    clinical_interpretation_allowed = (
        rule_allows_clinical_interpretation
        and bool(pred_safety["clinical_interpretation_allowed"])
        and bool(input_quality_result.clinical_interpretation_allowed)
    )

    # -------------------------------------------------------------------
    # Observed / Missing / Imputed audit (section 5)
    # -------------------------------------------------------------------
    raw_observation_status = _build_raw_observation_status(patient_record, schema)
    observed_count = sum(1 for v in raw_observation_status.values() if v == "OBSERVED")
    missing_count = sum(1 for v in raw_observation_status.values() if v == "MISSING")

    contribution_audit: list[dict[str, Any]] = []
    contribution_audit_note = None
    total_transformed_features = explainability_result.as_dict["contributions"]["count"]
    if expl_dict["explanation_computed"]:
        all_top = (
            explainability_result.as_dict["contributions"]["top_positive"]
            + explainability_result.as_dict["contributions"]["top_negative"]
        )
        for c in all_top[:CONTRIBUTION_AUDIT_LIMIT]:
            raw_feature = _raw_feature_from_transformed_name(c["transformed_feature"], schema)
            raw_status = raw_observation_status.get(raw_feature, "UNKNOWN")
            contribution_status = "OBSERVED" if raw_status == "OBSERVED" else "IMPUTED"
            contribution_audit.append(
                {
                    "transformed_feature": c["transformed_feature"],
                    "feature_label": c["feature_label"],
                    "raw_feature": raw_feature,
                    "raw_input_status": raw_status,
                    "contribution_value_status": contribution_status,
                    "contribution": c["contribution"],
                    "direction": c["direction"],
                }
            )
        imputed_in_top = sum(1 for c in contribution_audit if c["contribution_value_status"] == "IMPUTED")
        contribution_audit_note = (
            f"TOP-CONTRIBUTION AUDIT (not exhaustive): this audits only the "
            f"{len(contribution_audit)} highest-ranked contributions already "
            f"selected by the Explainability Agent, out of "
            f"{total_transformed_features} total transformed features. It "
            f"does not cover every transformed feature the model used."
            + (
                f" {imputed_in_top} of these {len(contribution_audit)} audited "
                "contributions are associated with raw features that were "
                "not directly observed in the submitted patient input and "
                "were handled by the locked preprocessing pipeline's "
                "fitted imputation. These contributions must not be "
                "interpreted as observed patient measurements."
                if imputed_in_top > 0
                else " All audited contributions in this top-ranked set "
                "correspond to raw features that were directly observed in "
                "the submitted patient input."
            )
        )
    else:
        contribution_audit_note = (
            "No explanation was computed for this input, so no "
            "top-contribution observed/imputed audit applies."
        )

    input_observation_summary = {
        "raw_feature_status": raw_observation_status,
        "observed_count": observed_count,
        "missing_count": missing_count,
        "total_raw_features": len(schema.locked_features),
        "contribution_observation_audit": contribution_audit,
        "contribution_audit_scope": "TOP_CONTRIBUTION_AUDIT_NOT_EXHAUSTIVE",
        "contribution_audit_limit": CONTRIBUTION_AUDIT_LIMIT,
        "total_transformed_features": total_transformed_features,
        "contribution_audit_note": contribution_audit_note,
    }

    # -------------------------------------------------------------------
    # Upstream statuses, preserved verbatim (not renamed)
    # -------------------------------------------------------------------
    upstream_statuses = {
        "input_quality_safety_gate": iq_decision["safety_gate"],
        "input_quality_data_quality_status": input_quality_result.as_dict["data_quality"]["quality_status"],
        "prediction_computed": pred_dict["prediction_computed"],
        "prediction_interpretation_status": pred_safety["interpretation_status"],
        "explanation_computed": expl_dict["explanation_computed"],
        "explanation_scope": explainability_result.as_dict["safety_inheritance"]["explanation_scope"],
        "trust_fairness_live_status": trust_fairness_result.live_evaluation_status,
        "trust_fairness_static_status": trust_fairness_result.static_reference_status,
    }

    population_evidence_status = {
        "current_session_status": trust_fairness_result.live_evaluation_status,
        "static_reference_status": trust_fairness_result.static_reference_status,
        "population_data_available": trust_fairness_result.as_dict["population_data_available"],
        "boundaries": TRUST_FAIRNESS_BOUNDARIES,
    }

    explanation_interpretation_status = {
        "explanation_method": explainability_result.as_dict.get("explanation_method"),
        "technical_disclaimer": explainability_result.as_dict.get("technical_disclaimer"),
        "boundaries": EXPLANATION_INTERPRETATION_BOUNDARIES,
    }

    safety_flags = {
        "model_trained": False,
        "preprocessor_fit": False,
        "model_modified": False,
        "preprocessor_modified": False,
        "threshold_modified": False,
        "schema_modified": False,
        "population_metrics_recomputed": False,
        "missing_values_fabricated": False,
        "patient_values_imputed_by_this_agent": False,
        "diagnosis_performed": False,
        "treatment_recommended": False,
        "medical_validation_claimed": False,
        "clinical_deployment_claimed": False,
        "upstream_restriction_loosened": False,
        "patient_level_fairness_claim_made": False,
        "imputed_values_presented_as_observed": False,
    }

    clinical_gate_disclaimer = (
        "Clinical interpretation gate passed for this demonstrator because "
        "all required clinical measurements are present. This does not "
        "constitute diagnosis, medical advice, or clinical validation."
        if safety_decision == "CLINICAL_INTERPRETATION_GATE_PASSED"
        else None
    )

    audit_summary = (
        f"safety_decision={safety_decision}; "
        f"technical_prediction_allowed={technical_prediction_allowed}; "
        f"technical_explanation_allowed={technical_explanation_allowed}; "
        f"clinical_interpretation_allowed={clinical_interpretation_allowed}; "
        f"clinical_measurements_observed={clinical_observed_count}/{clinical_total}; "
        f"raw_features_observed={observed_count}/{len(schema.locked_features)}; "
        f"reason_codes={reason_codes}"
    )

    result_dict = {
        "agent_name": "Safety_Agent",
        "phase": "Phase 6",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "safety_decision": safety_decision,
        "clinical_interpretation_allowed": clinical_interpretation_allowed,
        "technical_prediction_allowed": technical_prediction_allowed,
        "technical_explanation_allowed": technical_explanation_allowed,
        "clinical_gate_disclaimer": clinical_gate_disclaimer,
        "reason_codes": reason_codes,
        "safety_flags": safety_flags,
        "input_observation_summary": input_observation_summary,
        "upstream_statuses": upstream_statuses,
        "population_evidence_status": population_evidence_status,
        "explanation_interpretation_status": explanation_interpretation_status,
        "audit_summary": audit_summary,
    }

    return SafetyResult(as_dict=result_dict)

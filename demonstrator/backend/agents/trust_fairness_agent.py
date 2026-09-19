"""
Trust/Fairness Agent.

This agent has a fundamentally different character from the Prediction and
Explainability Agents: it does NOT run any computation on the current
patient's data. It never touches the locked model, the locked
preprocessor, or the patient's transformed feature representation. There
is no .transform(), no .predict_proba(), no coefficient arithmetic, and
therefore no possibility of fitting/refitting anything here.

It reports two clearly separated things:

  1. LIVE, CURRENT-SESSION population evaluation: this Streamlit
     demonstrator accepts one patient record at a time. One record is
     never a valid population sample, so this is ALWAYS reported as
     POPULATION_EVALUATION_UNAVAILABLE, regardless of what the current
     patient's data looks like or what the upstream agents concluded. No
     fairness metric is ever computed from the current session's input.

  2. STATIC POPULATION REFERENCE EVIDENCE: pre-validated, already-computed
     group-wise performance/fairness/calibration/explanation-consistency
     evidence from NB5 (the model's own held-out test cohort evaluation,
     n=6242 rows / 2939 unique participants) and NB6 (explanation
     consistency evidence for the same locked model). These values are
     read verbatim from CSV files copied unmodified from the NB5/NB6
     notebooks (see artifacts/evidence/) — nothing here is recomputed,
     estimated, or invented. Every number in this agent's output that
     isn't a sample count of the loaded evidence rows traces back to one
     of those two files.

Governance boundaries enforced throughout:
  - Population-level evidence is never presented as an individual-level
    guarantee for the current patient.
  - Equal or unequal metrics across groups are never described as
    "proving" fairness or its absence — only as descriptive findings.
  - No causal fairness claim, no clinical validity claim, no deployment-
    readiness claim.
  - Group sample sizes are always reported alongside metrics; a group
    documented in NB5 as having too few positive cases for stable
    inference (the under-18 age group, 5 positive cases) is explicitly
    flagged as insufficient rather than silently included as if reliable.
  - This agent does not use the Explainability Agent's contribution
    rankings as fairness evidence of any kind.
  - This agent inherits (does not recompute or override)
    clinical_interpretation_allowed from the upstream Prediction Agent
    result, if available. Population-level evidence being available NEVER
    flips this flag to True by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backend.agents.prediction_agent import PredictionResult
from backend.agents.input_quality_agent import InputQualityResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EVIDENCE_DIR = PROJECT_ROOT / "artifacts" / "evidence"
NB5_EVIDENCE_PATH = EVIDENCE_DIR / "nb5_trust_fairness_evidence.csv"
NB5_GROUP_DEFINITIONS_PATH = EVIDENCE_DIR / "nb5_fairness_group_definitions.csv"
NB6_EVIDENCE_PATH = EVIDENCE_DIR / "nb6_integrated_explanation_consistency_evidence.csv"

# Held-out test cohort this static evidence was computed on (NB5). This is
# reported for provenance only; it is not recomputed here.
NB5_TEST_COHORT_ROWS = 6242
NB5_TEST_COHORT_UNIQUE_PARTICIPANTS = 2939

# Groups NB5 itself documented as having too few positive cases for
# stable primary fairness inference, transcribed verbatim from NB5's own
# evidence table (dimension="Fairness — Age", indicator="<18
# positive-case count", value=5). This is not a threshold invented here —
# it is NB5's own stated conclusion about its own data.
INSUFFICIENT_DATA_GROUPS = {
    ("age_group", "<18"): (
        "NB5 documented only 5 positive cases in this subgroup within "
        "the held-out test cohort — too few for stable fairness "
        "inference. NB5 excluded this group from primary fairness "
        "conclusions for this reason."
    ),
}

DATA_LIMITATIONS = [
    "Observed subgroup disparities are descriptive evaluation findings, "
    "not proof of discrimination.",
    "These findings do not establish causality.",
    "These findings do not establish clinical harm.",
    "NHANES records may contain repeated observations from the same "
    "participants; subgroup analyses are treated as evaluation evidence "
    "rather than causal population claims.",
    "The under-18 age subgroup is not treated as a primary fairness "
    "conclusion because it contained only 5 positive cases in the NB5 "
    "evaluation.",
    "This evidence describes the locked model's performance on one "
    "held-out NHANES test cohort. It does not establish external "
    "validity, clinical utility, or deployment readiness.",
    "Explanation-consistency evidence (NB6) is a model-level "
    "trustworthiness signal only — it is not fairness evidence and is "
    "not used as such here.",
    "Calibration is imperfect; model probabilities must not be presented "
    "as calibrated clinical risk on the strength of this evidence alone.",
]


@dataclass(frozen=True)
class TrustFairnessResult:
    """Structured result of the Trust/Fairness Agent."""

    as_dict: dict[str, Any] = field(repr=False)

    @property
    def live_evaluation_status(self) -> str:
        return self.as_dict["evaluation_scope"]["current_session"]["evaluation_status"]

    @property
    def static_reference_status(self) -> str:
        return self.as_dict["evaluation_scope"]["static_population_reference"]["evaluation_status"]

    @property
    def clinical_interpretation_allowed(self) -> bool:
        return self.as_dict["clinical_interpretation_allowed"]


def _load_static_population_evidence() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]
]:
    """
    Load the three static, pre-validated evidence CSVs. Returns
    (metrics_rows, group_rows, explanation_consistency_rows, load_errors).
    Never raises — any load problem is captured in load_errors so the
    agent can report FAIRNESS_EVALUATION_BLOCKED honestly instead of
    crashing.
    """
    load_errors: list[str] = []

    metrics_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    consistency_rows: list[dict[str, Any]] = []

    try:
        metrics_rows = pd.read_csv(NB5_EVIDENCE_PATH).to_dict(orient="records")
    except Exception as exc:  # noqa: BLE001
        load_errors.append(f"Failed to load NB5 trust/fairness evidence: {exc}")

    try:
        group_rows = pd.read_csv(NB5_GROUP_DEFINITIONS_PATH).to_dict(orient="records")
    except Exception as exc:  # noqa: BLE001
        load_errors.append(f"Failed to load NB5 group definitions: {exc}")

    try:
        consistency_rows = pd.read_csv(NB6_EVIDENCE_PATH).to_dict(orient="records")
    except Exception as exc:  # noqa: BLE001
        load_errors.append(f"Failed to load NB6 explanation-consistency evidence: {exc}")

    return metrics_rows, group_rows, consistency_rows, load_errors


def run_trust_fairness_agent(
    input_quality_result: InputQualityResult | None,
    prediction_result: PredictionResult | None,
) -> TrustFairnessResult:
    """
    Run the Trust/Fairness Agent.

    Unlike the Prediction and Explainability Agents, this agent does not
    require a valid prediction to run — the static population reference
    evidence is independent of the current session's patient input and is
    always reportable (or, on a load failure, always reportable as
    blocked). input_quality_result and prediction_result are consumed
    only to (a) inherit clinical_interpretation_allowed without
    recomputing or overriding it, and (b) record, for transparency, that
    this evaluation is independent of this session's patient-level
    outcome.
    """
    metrics_rows, group_rows, consistency_rows, load_errors = _load_static_population_evidence()

    static_evidence_available = len(load_errors) == 0 and len(metrics_rows) > 0 and len(group_rows) > 0

    # -------------------------------------------------------------------
    # Current-session (live) evaluation scope: always unavailable. One
    # patient record is never a valid population sample.
    # -------------------------------------------------------------------
    current_session_block = {
        "evaluation_status": "POPULATION_EVALUATION_UNAVAILABLE",
        "reason": (
            "This Streamlit demonstrator evaluates one patient record at "
            "a time. A single individual input is insufficient for "
            "population-level fairness evaluation. No fairness metric "
            "has been calculated from this session's patient data, and "
            "none ever will be — this is a structural limitation of "
            "single-record evaluation, not a data-quality issue."
        ),
        "records_in_this_session": 1,
    }

    # -------------------------------------------------------------------
    # Static population reference evidence: pre-validated NB5/NB6 data,
    # independent of the current session.
    # -------------------------------------------------------------------
    if not static_evidence_available:
        static_block = {
            "evaluation_status": "FAIRNESS_EVALUATION_BLOCKED",
            "reason": "Static population reference evidence could not be loaded.",
            "load_errors": load_errors,
            "metrics": {},
            "group_definitions": [],
            "explanation_consistency": [],
        }
    else:
        # Group evidence by dimension for readability, verbatim values.
        metrics_by_dimension: dict[str, list[dict[str, Any]]] = {}
        for row in metrics_rows:
            metrics_by_dimension.setdefault(row["dimension"], []).append(row)

        # Flag groups NB5 itself documented as insufficient, verbatim.
        annotated_groups = []
        for row in group_rows:
            key = (row["group_variable"], row["group_label"])
            annotated = dict(row)
            if key in INSUFFICIENT_DATA_GROUPS:
                annotated["insufficient_data"] = True
                annotated["insufficient_data_note"] = INSUFFICIENT_DATA_GROUPS[key]
            else:
                annotated["insufficient_data"] = False
                annotated["insufficient_data_note"] = None
            annotated_groups.append(annotated)

        any_insufficient = any(g["insufficient_data"] for g in annotated_groups)

        static_block = {
            "evaluation_status": (
                "INSUFFICIENT_GROUP_DATA" if any_insufficient else "POPULATION_EVALUATION_AVAILABLE"
            ),
            "reason": (
                "Pre-validated population-level evidence from the locked "
                "model's held-out test cohort evaluation (NB5) and "
                "explanation-consistency evaluation (NB6) is available. "
                "At least one demographic subgroup has too few positive "
                "cases for stable inference; see group_definitions."
                if any_insufficient
                else "Pre-validated population-level evidence from the "
                "locked model's held-out test cohort evaluation (NB5) "
                "and explanation-consistency evaluation (NB6) is "
                "available."
            ),
            "test_cohort_rows": NB5_TEST_COHORT_ROWS,
            "test_cohort_unique_participants": NB5_TEST_COHORT_UNIQUE_PARTICIPANTS,
            "metrics": metrics_by_dimension,
            "group_definitions": annotated_groups,
            "explanation_consistency": consistency_rows,
            "provenance": {
                "source_notebooks": ["NB5", "NB6"],
                "source_files": [
                    str(NB5_EVIDENCE_PATH.relative_to(PROJECT_ROOT)),
                    str(NB5_GROUP_DEFINITIONS_PATH.relative_to(PROJECT_ROOT)),
                    str(NB6_EVIDENCE_PATH.relative_to(PROJECT_ROOT)),
                ],
                "values_recomputed_in_this_application": False,
                "note": (
                    "Every value in this block was copied verbatim from "
                    "the NB5/NB6 notebooks' own executed evidence tables "
                    "into the CSV files listed above. This application "
                    "performs no statistical computation to produce them."
                ),
            },
        }

    # -------------------------------------------------------------------
    # Inherit (never recompute or override) clinical_interpretation_allowed
    # -------------------------------------------------------------------
    if prediction_result is not None and prediction_result.prediction_computed:
        inherited_clinical_interpretation_allowed = prediction_result.clinical_interpretation_allowed
        inherited_interpretation_status = prediction_result.interpretation_status
    else:
        inherited_clinical_interpretation_allowed = False
        inherited_interpretation_status = (
            "PREDICTION_NOT_AVAILABLE"
            if prediction_result is not None
            else "PREDICTION_AGENT_NOT_YET_RUN"
        )

    governance_flags = {
        "model_retrained": False,
        "preprocessor_refit": False,
        "threshold_modified": False,
        "schema_modified": False,
        "processed_representation_modified": False,
        "missing_values_fabricated": False,
        "sensitive_attributes_inferred": False,
        "causal_fairness_claimed": False,
        "clinical_validity_claimed": False,
        "deployment_readiness_claimed": False,
        "individual_fairness_guaranteed": False,
        "equal_metrics_claimed_as_proof_of_fairness": False,
        "population_claim_made_from_single_patient": False,
        "explanation_contributions_used_as_fairness_evidence": False,
        "safety_override_performed": False,
    }

    result_dict = {
        "agent_name": "Trust_Fairness_Agent",
        "phase": "Phase 5",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "status": "EXECUTED",
        "population_data_available": static_evidence_available,
        "evaluation_scope": {
            "current_session": current_session_block,
            "static_population_reference": static_block,
        },
        "data_limitations": DATA_LIMITATIONS,
        "governance_flags": governance_flags,
        "inherited_from_upstream_agents": {
            "clinical_interpretation_allowed": inherited_clinical_interpretation_allowed,
            "interpretation_status": inherited_interpretation_status,
            "note": (
                "This value is inherited unchanged from the Prediction "
                "Agent's safety_inheritance. This agent never sets it to "
                "True and never overrides an upstream safety decision — "
                "population-level evidence being available does not by "
                "itself enable clinical interpretation."
            ),
        },
        "clinical_interpretation_allowed": inherited_clinical_interpretation_allowed,
        "summary_statement": (
            "Population-level fairness evaluation is not available from "
            "a single individual input. No fairness metric has been "
            "calculated for this session's patient. Pre-validated "
            "population-level trust and fairness evidence from the "
            "locked model's held-out evaluation is shown separately "
            "below for reference and is not a statement about this "
            "individual."
        ),
    }

    return TrustFairnessResult(as_dict=result_dict)

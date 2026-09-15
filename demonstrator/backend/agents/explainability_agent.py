"""
Explainability Agent.

Implements the exact explanation method established in NB6: an additive
decomposition of the locked Logistic Regression model's logit into
per-transformed-feature contributions.

    contribution_i = logistic_coefficient_i * transformed_feature_value_i
    logit = intercept + sum(contribution_i for all i)
    probability = sigmoid(logit)

This is a deterministic, closed-form property of a linear (logistic)
model — it is NOT an approximation, a sampling-based method, and it is
NOT SHAP. It must never be described as SHAP anywhere in this codebase.
The terms "SHAP values", "SHAP explanation", "SHAP contribution", and
"SHAP feature importance" do not apply to this method and are not used.

This agent does NOT:
  - fit, refit, or modify the locked model or preprocessor
    (preprocessor.transform is used; .fit()/.fit_transform() are never
    called anywhere in this module)
  - build a second, independent preprocessing pipeline — it reuses the
    exact same input-construction helper the Prediction Agent uses
    (backend.agents.prediction_agent._build_locked_input_frame), so the
    transformed representation is guaranteed identical to Phase 3's
    representation, not merely similar
  - modify the locked decision threshold
  - fabricate, invent, or silently impute any missing patient value
  - claim a causal relationship, clinical validity, or that an individual
    explanation is medically correct
  - invent "clinical importance" labels — a contribution's sign only
    describes which class it pushed the model output toward, and its
    magnitude only describes the model's own arithmetic, not clinical
    significance

A positive contribution means the transformed feature pushed the model
output toward MODEL_CLASS_1. A negative contribution means it pushed the
model output toward MODEL_CLASS_0. This is a model contribution, not a
clinical causal effect.

Governance / safety-gate handling:
  - If the Prediction Agent did not compute a prediction (i.e.
    prediction.prediction_computed == False, which happens when
    safety_gate == INVALID_INPUT_SCHEMA), this agent does NOT run at all.
    No transform() or coefficient arithmetic is attempted.
  - If a prediction was computed but clinical_interpretation_allowed ==
    False (missing/partial clinical measurements), this agent DOES
    compute the technical decomposition — consistent with the Prediction
    Agent's own behavior in that same situation — but the result is
    always and unconditionally labeled a technical/model-level
    explanation, never a clinical interpretation, regardless of the
    inherited flag's value.

This agent inherits the Input/Data Quality Agent's and Prediction
Agent's safety flags verbatim (via the already-computed
InputQualityResult and PredictionResult) rather than recomputing or
overriding them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from backend.governance import Governance
from backend.model_loader import ModelBundle
from backend.schema import LockedSchema
from backend.agents.input_quality_agent import InputQualityResult
from backend.agents.prediction_agent import PredictionResult, _build_locked_input_frame

EXPLANATION_METHOD = "Logistic coefficient contribution decomposition"

TECHNICAL_DISCLAIMER = (
    "This explanation describes how transformed input features "
    "contributed to the locked model output. It is not a causal "
    "explanation, medical diagnosis, or clinical recommendation."
)

# Number of top positive / top negative contributions surfaced in the
# structured result and the UI. All 517 contributions are still computed
# and available via `contributions.count`; this only limits what is
# highlighted as "top" for display purposes.
TOP_K = 15

# Reconstruction tolerance for the additive-logit check. Observed
# float64 reconstruction error for this model is at machine-epsilon
# scale (~1e-15). This tolerance is intentionally far looser than that
# to remain robust to minor BLAS/platform floating-point variation while
# still being a meaningful correctness check — not a way to paper over a
# real discrepancy.
RECONSTRUCTION_TOLERANCE = 1e-9


@dataclass(frozen=True)
class ExplainabilityResult:
    """Structured result of the Explainability Agent."""

    as_dict: dict[str, Any] = field(repr=False)

    @property
    def explanation_computed(self) -> bool:
        return self.as_dict["explanation"]["explanation_computed"]

    @property
    def reconstruction_error(self) -> float | None:
        return self.as_dict["explanation"]["reconstruction_error"]

    @property
    def reconstructed_probability(self) -> float | None:
        return self.as_dict["explanation"]["reconstructed_probability"]

    @property
    def top_positive_contributions(self) -> list[dict[str, Any]]:
        return self.as_dict["contributions"]["top_positive"]

    @property
    def top_negative_contributions(self) -> list[dict[str, Any]]:
        return self.as_dict["contributions"]["top_negative"]


def _humanize_transformed_name(transformed_name: str) -> str:
    """
    Strip the ColumnTransformer prefix ('numeric__' or 'categorical__')
    from a transformed feature name to get a more readable label. This
    uses only the preprocessor's own naming convention verbatim — it does
    not invent, reinterpret, or relabel anything.
    """
    for prefix in ("numeric__", "categorical__"):
        if transformed_name.startswith(prefix):
            return transformed_name[len(prefix):]
    return transformed_name


def run_explainability_agent(
    patient_record: dict[str, Any],
    schema: LockedSchema,
    model_bundle: ModelBundle,
    governance: Governance,
    input_quality_result: InputQualityResult,
    prediction_result: PredictionResult,
) -> ExplainabilityResult:
    """
    Run the Explainability Agent for one patient record.

    model_bundle must already have is_safe_to_use == True. This function
    does not re-check artifact integrity.
    """
    pred = prediction_result.as_dict["prediction"]
    safety_inheritance = prediction_result.as_dict["safety_inheritance"]

    base_governance_block = {
        "model_retrained": False,
        "preprocessor_refit": False,
        "threshold_modified": False,
        "missing_values_fabricated": False,
        "patient_values_manually_imputed": False,
        "clinical_measurements_invented": False,
        "clinical_importance_labels_invented": False,
        "causal_relationship_claimed": False,
        "clinical_validity_claimed": False,
        "explanation_method_is_shap": False,
        "diagnosis_performed": False,
        "treatment_recommended": False,
        "safety_override_performed": False,
    }

    if not pred["prediction_computed"]:
        # Hard gate: the Prediction Agent did not compute a prediction
        # (invalid input schema). Do not run any explanation logic.
        result_dict = {
            "agent": "Explainability_Agent",
            "agent_version": "PHASE4_EXPLAINABILITY_AGENT_V1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "research_only": True,
            "explanation_method": EXPLANATION_METHOD,
            "technical_disclaimer": TECHNICAL_DISCLAIMER,
            "explanation": {
                "explanation_computed": False,
                "reason": "Prediction Agent did not compute a prediction for this input.",
                "intercept": None,
                "total_logit": None,
                "reconstructed_logit": None,
                "reconstruction_error": None,
                "reconstructed_probability": None,
                "threshold": model_bundle.locked_threshold,
                "predicted_class": None,
            },
            "contributions": {
                "top_positive": [],
                "top_negative": [],
                "count": 0,
            },
            "safety_inheritance": {
                **safety_inheritance,
                "explanation_scope": "NOT_COMPUTED",
            },
            "governance": base_governance_block,
        }
        return ExplainabilityResult(as_dict=result_dict)

    # -------------------------------------------------------------------
    # Rebuild the exact same transformed representation the Prediction
    # Agent used. Reuses the same helper function — no second pipeline.
    # -------------------------------------------------------------------
    X_input = _build_locked_input_frame(patient_record, schema)
    X_transformed = model_bundle.preprocessor.transform(X_input)  # transform only, no fit

    X_arr = np.asarray(
        X_transformed.todense() if hasattr(X_transformed, "todense") else X_transformed
    )[0]

    feature_names = list(model_bundle.preprocessor.get_feature_names_out())

    if len(feature_names) != X_arr.shape[0]:
        raise ValueError(
            "Transformed feature count does not match preprocessor's "
            "reported feature names; refusing to produce an explanation."
        )

    coef = model_bundle.model.coef_[0]
    intercept = float(model_bundle.model.intercept_[0])

    if coef.shape[0] != X_arr.shape[0]:
        raise ValueError(
            "Model coefficient count does not match transformed feature "
            "count; refusing to produce an explanation."
        )

    contributions_arr = coef * X_arr
    reconstructed_logit = float(intercept + contributions_arr.sum())

    # Reference logit computed directly by the locked model itself
    # (decision_function), used only to numerically verify the additive
    # reconstruction above — not a second, independently-derived value.
    total_logit = float(model_bundle.model.decision_function(X_transformed)[0])

    reconstruction_error = abs(reconstructed_logit - total_logit)

    reconstructed_probability = float(1.0 / (1.0 + np.exp(-reconstructed_logit)))
    predicted_class = int(reconstructed_probability >= model_bundle.locked_threshold)

    # Probability/class consistency against the Prediction Agent's own
    # stored result (same model, same input — should match essentially
    # exactly; reported explicitly rather than assumed).
    probability_consistency_error = abs(
        reconstructed_probability - pred["predicted_probability"]
    )
    class_consistent_with_prediction_agent = predicted_class == pred["predicted_class"]

    all_contributions = [
        {
            "transformed_feature": name,
            "feature_label": _humanize_transformed_name(name),
            "contribution": float(value),
            "direction": (
                "TOWARD_MODEL_CLASS_1"
                if value > 0
                else "TOWARD_MODEL_CLASS_0"
                if value < 0
                else "NEUTRAL"
            ),
        }
        for name, value in zip(feature_names, contributions_arr)
    ]

    top_positive = sorted(
        [c for c in all_contributions if c["contribution"] > 0],
        key=lambda c: c["contribution"],
        reverse=True,
    )[:TOP_K]
    top_negative = sorted(
        [c for c in all_contributions if c["contribution"] < 0],
        key=lambda c: c["contribution"],
    )[:TOP_K]

    if safety_inheritance["clinical_interpretation_allowed"]:
        explanation_scope = "TECHNICAL_MODEL_EXPLANATION_ONLY"
    else:
        explanation_scope = "TECHNICAL_MODEL_EXPLANATION_ONLY_CLINICAL_INTERPRETATION_BLOCKED"

    result_dict = {
        "agent": "Explainability_Agent",
        "agent_version": "PHASE4_EXPLAINABILITY_AGENT_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "research_only": True,
        "explanation_method": EXPLANATION_METHOD,
        "technical_disclaimer": TECHNICAL_DISCLAIMER,
        "explanation": {
            "explanation_computed": True,
            "intercept": intercept,
            "total_logit": total_logit,
            "reconstructed_logit": reconstructed_logit,
            "reconstruction_error": reconstruction_error,
            "reconstruction_within_tolerance": reconstruction_error <= RECONSTRUCTION_TOLERANCE,
            "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
            "reconstructed_probability": reconstructed_probability,
            "prediction_agent_probability": pred["predicted_probability"],
            "probability_consistency_error": probability_consistency_error,
            "threshold": model_bundle.locked_threshold,
            "predicted_class": predicted_class,
            "predicted_class_consistent_with_prediction_agent": class_consistent_with_prediction_agent,
        },
        "contributions": {
            "top_positive": top_positive,
            "top_negative": top_negative,
            "count": len(all_contributions),
        },
        "safety_inheritance": {
            **safety_inheritance,
            "explanation_scope": explanation_scope,
        },
        "governance": base_governance_block,
    }

    return ExplainabilityResult(as_dict=result_dict)

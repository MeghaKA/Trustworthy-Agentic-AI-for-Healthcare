"""
Agentic Orchestrator (Phase 8).

This is a coordination layer only. It calls the six existing, already-
locked agent functions in their established order, using their existing
public interfaces exactly as-is:

    run_input_quality_agent   (Phase 2)
    run_prediction_agent      (Phase 3)
    run_explainability_agent  (Phase 4)
    run_trust_fairness_agent  (Phase 5)
    run_safety_agent          (Phase 6)
    run_cds_reporting_agent   (Phase 7)

The orchestrator does NOT:
  - perform model inference, preprocessing, explainability arithmetic,
    fairness computation, or clinical interpretation itself
  - call .fit(), .fit_transform(), or otherwise train/refit anything
  - implement a second prediction, explanation, or reporting pipeline
  - make any LLM call, network call, or external API call
  - introduce randomness, hidden state, or dynamic threshold/schema
    selection
  - loosen any restriction a downstream agent (in particular the Safety
    Agent) has already imposed

It is a plain, deterministic Python function: the same patient_record and
the same locked artifacts/evidence always produce the same orchestration
result. There is no autonomous decision-making here — every governance
decision is made by the existing Phase 2-7 agents; this module only
calls them in order and packages what they returned.

GOVERNED BLOCKS ARE NOT ORCHESTRATION FAILURES.
INVALID_INPUT_SCHEMA, 0/7 clinical measurements, and partial clinical
measurements are all *normal, expected* governed outcomes that the
existing agents already handle internally (e.g. the Prediction Agent
already refuses to call preprocessor.transform()/model.predict_proba()
on an invalid schema, without raising an exception). The orchestrator
does not special-case these — it runs every stage in the fixed order
regardless, exactly as app.py already did before this phase, and each
agent's own result object correctly encodes the block. This preserves
the exact existing Phase 1-7 behavior rather than reimplementing it.

ORCHESTRATION FAILURE is reserved for a genuine, unexpected exception
raised by one of the six agent calls (a real bug, not a governed block).
In that case the orchestrator stops calling further stages, records the
failure, and returns a result that cannot claim any permission that was
never actually established — no fallback clinical output is invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from backend.governance import Governance
from backend.model_loader import ModelBundle
from backend.schema import LockedSchema

from backend.agents.input_quality_agent import InputQualityResult, run_input_quality_agent
from backend.agents.prediction_agent import PredictionResult, run_prediction_agent
from backend.agents.explainability_agent import ExplainabilityResult, run_explainability_agent
from backend.agents.trust_fairness_agent import TrustFairnessResult, run_trust_fairness_agent
from backend.agents.safety_agent import SafetyResult, run_safety_agent
from backend.agents.cds_reporting_agent import CDSReportingResult, run_cds_reporting_agent

# Fixed execution order. This is the single source of truth for stage
# order in this module; nothing here reorders or skips a stage based on
# input content (only an unexpected exception can shorten the trace).
EXECUTION_STAGES: tuple[str, ...] = (
    "input_quality",
    "prediction",
    "explainability",
    "trust_fairness",
    "safety",
    "cds_reporting",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class OrchestrationResult:
    """
    Structured result of one end-to-end orchestration run.

    The individual agent result objects (input_quality_result,
    prediction_result, etc.) are the SAME objects the six existing agents
    already produce — nothing here re-wraps, re-derives, or mutates them.
    Any of these may be None if orchestration stopped early due to an
    unexpected exception in an earlier stage.
    """

    input_quality_result: InputQualityResult | None
    prediction_result: PredictionResult | None
    explainability_result: ExplainabilityResult | None
    trust_fairness_result: TrustFairnessResult | None
    safety_result: SafetyResult | None
    cds_reporting_result: CDSReportingResult | None

    execution_trace: tuple[dict[str, Any], ...]
    overall_status: str  # "COMPLETED" | "ORCHESTRATION_FAILED"
    workflow_status: str  # safety_result.safety_decision, or a failure marker
    governance_errors: tuple[str, ...]
    timestamp_utc: str

    @property
    def as_dict(self) -> dict[str, Any]:
        """
        JSON-safe structured orchestration summary: execution order, agent
        statuses, upstream results (as each agent's own as_dict — not
        re-derived), final Safety Agent status, final CDS/reporting
        result, overall workflow status, and any governance errors.
        """
        return {
            "orchestrator": "Agentic_Orchestrator",
            "phase": "Phase 8",
            "timestamp_utc": self.timestamp_utc,
            "research_only": True,
            "execution_order": list(EXECUTION_STAGES),
            "execution_trace": list(self.execution_trace),
            "overall_status": self.overall_status,
            "workflow_status": self.workflow_status,
            "governance_errors": list(self.governance_errors),
            "upstream_results": {
                "input_quality": self.input_quality_result.as_dict if self.input_quality_result else None,
                "prediction": self.prediction_result.as_dict if self.prediction_result else None,
                "explainability": self.explainability_result.as_dict if self.explainability_result else None,
                "trust_fairness": self.trust_fairness_result.as_dict if self.trust_fairness_result else None,
            },
            "final_safety_status": self.safety_result.as_dict if self.safety_result else None,
            "final_cds_report": self.cds_reporting_result.as_dict if self.cds_reporting_result else None,
            "governance_notes": [
                "This orchestrator performs no model inference, preprocessing, "
                "explainability computation, fairness computation, or clinical "
                "interpretation of its own. Every value above was produced by "
                "the existing, unmodified Phase 2-7 agent it is attributed to.",
                "The orchestrator cannot loosen a restriction any downstream "
                "agent (in particular the Safety Agent) has imposed — it only "
                "calls each agent once, in fixed order, and forwards what was "
                "returned.",
            ],
        }


def run_orchestrator(
    patient_record: dict[str, Any],
    schema: LockedSchema,
    model_bundle: ModelBundle,
    governance: Governance,
) -> OrchestrationResult:
    """
    Run the full six-agent workflow in order:

        input_quality -> prediction -> explainability -> trust_fairness
            -> safety -> cds_reporting

    model_bundle must already have is_safe_to_use == True (the caller is
    expected to have blocked the whole application already otherwise,
    exactly as app.py did before Phase 8 — this function does not
    re-check artifact integrity).
    """
    trace: list[dict[str, Any]] = []
    errors: list[str] = []

    def record_success(stage: str) -> None:
        trace.append({"stage": stage, "status": "SUCCESS", "error": None, "timestamp_utc": _now()})

    def record_failure(stage: str, exc: Exception) -> None:
        # Record that the stage failed without leaking an internal stack
        # trace to the user-facing application — just the stage name and
        # exception type/message.
        message = f"{type(exc).__name__}: {exc}"
        trace.append({"stage": stage, "status": "FAILED", "error": message, "timestamp_utc": _now()})
        errors.append(f"Unexpected exception in '{stage}' stage: {message}")

    def finalize(
        input_quality_result: InputQualityResult | None,
        prediction_result: PredictionResult | None,
        explainability_result: ExplainabilityResult | None,
        trust_fairness_result: TrustFairnessResult | None,
        safety_result: SafetyResult | None,
        cds_reporting_result: CDSReportingResult | None,
    ) -> OrchestrationResult:
        overall_status = "ORCHESTRATION_FAILED" if errors else "COMPLETED"
        if safety_result is not None:
            workflow_status = safety_result.safety_decision
        else:
            # No safety decision could be established at all -> the
            # workflow cannot claim any permission. This is the fail-safe
            # marker, never a fabricated "allowed" state.
            workflow_status = "INCOMPLETE_NO_SAFETY_DECISION_ESTABLISHED"
        return OrchestrationResult(
            input_quality_result=input_quality_result,
            prediction_result=prediction_result,
            explainability_result=explainability_result,
            trust_fairness_result=trust_fairness_result,
            safety_result=safety_result,
            cds_reporting_result=cds_reporting_result,
            execution_trace=tuple(trace),
            overall_status=overall_status,
            workflow_status=workflow_status,
            governance_errors=tuple(errors),
            timestamp_utc=_now(),
        )

    # -------------------------------------------------------------------
    # Stage 1: Input/Data Quality Agent (Phase 2, unmodified)
    # -------------------------------------------------------------------
    try:
        input_quality_result = run_input_quality_agent(patient_record, schema)
        record_success("input_quality")
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any agent failure must be caught and reported safely, never propagate as an unhandled crash
        record_failure("input_quality", exc)
        return finalize(None, None, None, None, None, None)

    # -------------------------------------------------------------------
    # Stage 2: Prediction Agent (Phase 3, unmodified). This call already
    # internally refuses to run model inference on INVALID_INPUT_SCHEMA;
    # the orchestrator does not duplicate or re-decide that gate.
    # -------------------------------------------------------------------
    try:
        prediction_result = run_prediction_agent(
            patient_record, schema, model_bundle, governance, input_quality_result
        )
        record_success("prediction")
    except Exception as exc:  # noqa: BLE001
        record_failure("prediction", exc)
        return finalize(input_quality_result, None, None, None, None, None)

    # -------------------------------------------------------------------
    # Stage 3: Explainability Agent (Phase 4, unmodified). Already
    # internally refuses to run when the Prediction Agent did not compute
    # a prediction.
    # -------------------------------------------------------------------
    try:
        explainability_result = run_explainability_agent(
            patient_record, schema, model_bundle, governance, input_quality_result, prediction_result
        )
        record_success("explainability")
    except Exception as exc:  # noqa: BLE001
        record_failure("explainability", exc)
        return finalize(input_quality_result, prediction_result, None, None, None, None)

    # -------------------------------------------------------------------
    # Stage 4: Trust/Fairness Agent (Phase 5, unmodified). Uses the
    # existing static NB5/NB6 evidence-driven implementation only.
    # -------------------------------------------------------------------
    try:
        trust_fairness_result = run_trust_fairness_agent(input_quality_result, prediction_result)
        record_success("trust_fairness")
    except Exception as exc:  # noqa: BLE001
        record_failure("trust_fairness", exc)
        return finalize(input_quality_result, prediction_result, explainability_result, None, None, None)

    # -------------------------------------------------------------------
    # Stage 5: Safety Agent (Phase 6, unmodified) — the final,
    # authoritative governance gate. If this stage itself fails, no
    # downstream stage may run: without a Safety result, no permission
    # can be considered established, so CDS/Reporting is not attempted.
    # -------------------------------------------------------------------
    try:
        safety_result = run_safety_agent(
            patient_record,
            schema,
            input_quality_result,
            prediction_result,
            explainability_result,
            trust_fairness_result,
        )
        record_success("safety")
    except Exception as exc:  # noqa: BLE001
        record_failure("safety", exc)
        return finalize(
            input_quality_result, prediction_result, explainability_result, trust_fairness_result, None, None
        )

    # -------------------------------------------------------------------
    # Stage 6: CDS/Reporting Agent (Phase 7, unmodified). Receives the
    # SAME safety_result object produced above — not a re-derived or
    # re-summarized copy — so the report cannot diverge from the
    # authoritative Safety Agent decision.
    # -------------------------------------------------------------------
    try:
        cds_reporting_result = run_cds_reporting_agent(
            governance,
            schema,
            input_quality_result,
            prediction_result,
            explainability_result,
            trust_fairness_result,
            safety_result,
        )
        record_success("cds_reporting")
    except Exception as exc:  # noqa: BLE001
        record_failure("cds_reporting", exc)
        return finalize(
            input_quality_result,
            prediction_result,
            explainability_result,
            trust_fairness_result,
            safety_result,
            None,
        )

    return finalize(
        input_quality_result,
        prediction_result,
        explainability_result,
        trust_fairness_result,
        safety_result,
        cds_reporting_result,
    )

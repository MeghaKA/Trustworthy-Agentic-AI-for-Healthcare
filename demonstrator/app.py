"""
Trustworthy Multi-Agent Healthcare AI Demonstrator — Streamlit app shell.

PHASES 1–4 IMPLEMENTED:
  - Phase 1: Persistent research-only banner, startup governance load,
    locked-artifact integrity verification.
  - Phase 2: Structured patient-input form (curated 30-field subset) and
    the Input/Data Quality Agent (exact NB9 Cell 14 logic).
  - Phase 3: The Prediction Agent (exact NB9 Cell 15 logic), chained
    immediately after the Input/Data Quality Agent.
  - Phase 4: The Explainability Agent (exact NB6 additive logistic
    coefficient contribution decomposition), chained immediately after
    the Prediction Agent.
  - Phase 5: The Trust/Fairness Agent (static, pre-validated NB5/NB6
    population-level evidence; live single-input evaluation always
    reported as unavailable), chained immediately after the
    Explainability Agent.
  - Phase 6: The Safety Agent (deterministic governance control point;
    inherits and can only tighten upstream flags; distinguishes
    observed/missing/imputed inputs), chained immediately after the
    Trust/Fairness Agent. Its flags now gate what the Prediction and
    Explainability sections display, so it cannot be bypassed through
    the normal UI workflow.

CDS/Reporting and the Orchestrator (Phases 7–8) are not yet implemented
and remain placeholders in the pipeline scaffold below.
"""

from __future__ import annotations

import streamlit as st

from backend.governance import Governance, GovernanceConfigError, load_governance
from backend.model_loader import ModelBundle, load_model_bundle
from backend.schema import LockedSchema, SchemaError, load_locked_schema
from backend.agents.input_quality_agent import InputQualityResult, run_input_quality_agent
from backend.agents.prediction_agent import PredictionResult, run_prediction_agent
from backend.agents.explainability_agent import ExplainabilityResult, run_explainability_agent
from backend.agents.trust_fairness_agent import TrustFairnessResult, run_trust_fairness_agent
from backend.agents.safety_agent import SafetyResult, run_safety_agent
from ui.patient_input_form import render_patient_input_form


st.set_page_config(
    page_title="Trustworthy Multi-Agent Healthcare AI Demonstrator",
    page_icon="🩺",
    layout="wide",
)


def render_banner(banner_text: str) -> None:
    """Persistent research-only banner. Rendered on every run, at the top."""
    st.markdown(
        f"""
        <div style="
            background-color:#7a1f1f;
            color:white;
            padding:0.6rem 1rem;
            border-radius:6px;
            font-weight:600;
            text-align:center;
            margin-bottom:1rem;
        ">
            ⚠️ {banner_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading governance configuration...")
def get_governance() -> Governance:
    return load_governance()


@st.cache_resource(show_spinner="Verifying and loading locked model artifacts...")
def get_model_bundle(_governance: Governance) -> ModelBundle:
    # Governance is hashable-unfriendly (contains dicts), so we prefix the
    # cache key arg with an underscore to tell Streamlit not to hash it;
    # cache invalidation for this resource is process-lifetime only, which
    # is correct for locked, immutable artifacts.
    return load_model_bundle(_governance)


@st.cache_resource(show_spinner="Loading locked 214-feature input schema...")
def get_schema() -> LockedSchema:
    return load_locked_schema()


def render_integrity_panel(governance: Governance, bundle: ModelBundle) -> None:
    with st.expander("🔒 Startup Governance & Artifact Integrity", expanded=not bundle.is_safe_to_use):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Model artifact**")
            st.write(f"Path: `{bundle.model_integrity.path.name}`")
            st.write(f"File found: {bundle.model_integrity.file_exists}")
            st.write(f"Expected sha256: `{bundle.model_integrity.expected_sha256}`")
            st.write(f"Actual sha256: `{bundle.model_integrity.actual_sha256}`")
            st.write(
                "Hash match: "
                + ("✅ MATCH" if bundle.model_integrity.hash_match else "❌ MISMATCH")
            )

        with col2:
            st.markdown("**Preprocessor artifact**")
            st.write(f"Path: `{bundle.preprocessor_integrity.path.name}`")
            st.write(f"File found: {bundle.preprocessor_integrity.file_exists}")
            st.write(f"Expected sha256: `{bundle.preprocessor_integrity.expected_sha256}`")
            st.write(f"Actual sha256: `{bundle.preprocessor_integrity.actual_sha256}`")
            st.write(
                "Hash match: "
                + ("✅ MATCH" if bundle.preprocessor_integrity.hash_match else "❌ MISMATCH")
            )

        st.markdown("---")
        st.write(
            f"scikit-learn active version: `{bundle.sklearn_version_active}` "
            f"(required: `{bundle.sklearn_version_required}`) — "
            + ("✅ match" if bundle.sklearn_version_matches else "❌ mismatch")
        )
        st.write(
            f"Locked decision threshold (sourced from NB3 metadata only): "
            f"**{governance.locked_threshold}**"
        )
        st.write(
            f"Threshold selection method: {governance.threshold_selection_method} "
            f"(target sensitivity {governance.target_sensitivity})"
        )
        st.write(
            f"Raw feature count: {governance.raw_feature_count} → "
            f"Processed feature count: {governance.processed_feature_count}"
        )

        if bundle.is_safe_to_use:
            st.success("Locked model and preprocessor verified and loaded successfully.")
        else:
            st.error("Locked model/preprocessor integrity verification FAILED.")
            for reason in bundle.failure_reasons:
                st.write(f"- {reason}")


def render_input_quality_result(result: InputQualityResult) -> None:
    r = result.as_dict

    st.success("Input/Data Quality Agent executed.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model-input completeness", f"{r['data_quality']['completeness']*100:.2f}%")
    col2.metric(
        "Clinical measurement completeness",
        f"{r['clinical_measurements']['completeness']*100:.2f}%",
    )
    col3.metric("Quality status", r["data_quality"]["quality_status"])

    st.write(
        f"**Safety gate:** `{r['decision']['safety_gate']}`  —  "
        f"prediction_input_valid=`{r['decision']['prediction_input_valid']}`, "
        f"clinical_interpretation_allowed=`{r['decision']['clinical_interpretation_allowed']}`"
    )

    if not r["decision"]["clinical_interpretation_allowed"]:
        st.warning(
            "Clinical interpretation is blocked for this input. This is "
            "expected, correct safety behavior when required information "
            "is missing or incomplete — it is not an error, and no value "
            "has been invented to work around it."
        )

    with st.expander("Full structured agent output (JSON contract)"):
        st.json(r)

    if r["clinical_measurements"]["missing"]:
        st.write("**Missing prediction-time clinical measurements:**")
        for feature in r["clinical_measurements"]["missing"]:
            st.write(f"- `{feature}`")

    st.caption(
        "Model-input completeness (over all 214 locked features) is "
        "distinct from clinical-measurement completeness (over the 7 "
        "prediction-time measurements only) — both are reported "
        "separately above, exactly as the underlying agent contract "
        "distinguishes them."
    )


def render_prediction_result(result: PredictionResult, safety_result: SafetyResult | None = None) -> None:
    r = result.as_dict

    if not r["prediction"]["prediction_computed"]:
        st.error(
            "Prediction Agent did not run. The input failed schema "
            "validation (invalid or unexpected fields), so no "
            "preprocessing or model computation was attempted. This is "
            "expected fail-safe behavior."
        )
        with st.expander("Full structured agent output (JSON contract)"):
            st.json(r)
        return

    # Safety Agent is the final, authoritative gate (Phase 6). It can only
    # tighten this section's own flags, never loosen them, but this check
    # ensures the Safety Agent's decision cannot be bypassed even if it
    # somehow disagreed with this agent's own flag.
    if safety_result is not None and not safety_result.technical_prediction_allowed:
        st.error(
            "Safety Agent has blocked technical prediction display for "
            f"this input (`{safety_result.safety_decision}`)."
        )
        with st.expander("Full structured agent output (JSON contract)"):
            st.json(r)
        return

    st.success("Prediction Agent executed.")

    st.caption(
        "The value below is a technical model output only. It is not a "
        "calibrated individual clinical risk estimate, not a clinical "
        "diagnosis, and not medical advice."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Model output (predicted probability)", f"{r['prediction']['predicted_probability']:.4f}")
    col2.metric("Locked threshold", f"{r['prediction']['threshold']:.2f}")
    col3.metric("Predicted class", r["prediction"]["prediction_label"])

    clinical_allowed = (
        safety_result.clinical_interpretation_allowed
        if safety_result is not None
        else r["safety_inheritance"]["clinical_interpretation_allowed"]
    )

    st.write(
        f"**Clinical interpretation allowed "
        f"(Safety Agent, Phase 6, authoritative):** `{clinical_allowed}`  —  "
        f"interpretation_status=`{r['safety_inheritance']['interpretation_status']}`"
    )

    if clinical_allowed:
        st.info(
            "This input was not blocked by the completeness checks used so "
            "far. That means data completeness alone does not prevent "
            "downstream components from proceeding — it does NOT mean this "
            "model output is clinically validated, medically interpretable, "
            "or suitable for diagnosis or treatment decisions. It remains a "
            "technical model output."
        )
    else:
        st.warning(
            "Clinical interpretation is blocked for this input (missing or "
            "incomplete prediction-time clinical measurements). A "
            "probability was still computed, exactly as the underlying "
            "agent contract specifies, but it must not be treated as a "
            "clinical interpretation of any kind."
        )

    with st.expander("Full structured agent output (JSON contract)"):
        st.json(r)

    st.caption(
        "Model class labels (`MODEL_CLASS_0` / `MODEL_CLASS_1`) are kept "
        "neutral and are not translated into diagnostic language."
    )


def render_explainability_result(result: ExplainabilityResult, safety_result: SafetyResult | None = None) -> None:
    r = result.as_dict

    if not r["explanation"]["explanation_computed"]:
        st.error(
            "Explainability Agent did not run. The Prediction Agent did "
            "not compute a prediction for this input, so no coefficient "
            "decomposition was attempted. This is expected fail-safe "
            "behavior."
        )
        with st.expander("Full structured agent output (JSON contract)"):
            st.json(r)
        return

    # Safety Agent is the final, authoritative gate (Phase 6).
    if safety_result is not None and not safety_result.technical_explanation_allowed:
        st.error(
            "Safety Agent has blocked technical explanation display for "
            f"this input (`{safety_result.safety_decision}`)."
        )
        with st.expander("Full structured agent output (JSON contract)"):
            st.json(r)
        return

    st.success("Explainability Agent executed.")
    st.write(f"**Explanation type:** {r['explanation_method']}")
    st.caption(r["technical_disclaimer"])
    st.caption(
        "Contributions shown below are model contributions only — "
        "coefficient × transformed feature value. They are not clinical "
        "risk factors, causes of disease, or treatment guidance."
    )

    e = r["explanation"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted probability", f"{e['reconstructed_probability']:.4f}")
    col2.metric("Threshold", f"{e['threshold']:.2f}")
    col3.metric("Predicted class", "MODEL_CLASS_1" if e["predicted_class"] == 1 else "MODEL_CLASS_0")
    col4.metric("Reconstruction error", f"{e['reconstruction_error']:.2e}")

    st.write(
        f"intercept = `{e['intercept']:.6f}`  ·  total logit = `{e['total_logit']:.6f}`  ·  "
        f"reconstructed logit = `{e['reconstructed_logit']:.6f}`"
    )
    st.write(
        f"Reconstruction within tolerance ({e['reconstruction_tolerance']:.0e}): "
        f"**{e['reconstruction_within_tolerance']}**  ·  "
        f"Predicted class consistent with Prediction Agent: "
        f"**{e['predicted_class_consistent_with_prediction_agent']}**"
    )

    clinical_allowed = (
        safety_result.clinical_interpretation_allowed
        if safety_result is not None
        else r["safety_inheritance"]["clinical_interpretation_allowed"]
    )
    if not clinical_allowed:
        st.warning(
            "Clinical interpretation remains blocked for this input "
            "(Safety Agent, Phase 6, authoritative). This is a technical "
            "model explanation only."
        )

    if safety_result is not None:
        note = safety_result.as_dict["input_observation_summary"]["contribution_audit_note"]
        if note:
            st.warning(note)

    st.markdown("**Top positive model contributions** (toward MODEL_CLASS_1)")
    pos = r["contributions"]["top_positive"]
    if pos:
        st.dataframe(
            [{"feature": c["feature_label"], "contribution": c["contribution"]} for c in pos],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("None.")

    st.markdown("**Top negative model contributions** (toward MODEL_CLASS_0)")
    neg = r["contributions"]["top_negative"]
    if neg:
        st.dataframe(
            [{"feature": c["feature_label"], "contribution": c["contribution"]} for c in neg],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("None.")

    st.caption(
        f"Showing top {len(pos)} positive and top {len(neg)} negative of "
        f"{r['contributions']['count']} total transformed-feature "
        "contributions. A large contribution reflects the model's own "
        "arithmetic only — it does not by itself indicate clinical "
        "importance."
    )

    with st.expander("Full structured agent output (JSON contract)"):
        st.json(r)


def render_trust_fairness_result(result: TrustFairnessResult) -> None:
    r = result.as_dict

    st.success("Trust/Fairness Agent executed.")
    st.write(f"**{r['summary_statement']}**")

    session = r["evaluation_scope"]["current_session"]
    static = r["evaluation_scope"]["static_population_reference"]

    st.markdown("**This session's input**")
    st.warning(f"`{session['evaluation_status']}` — {session['reason']}")

    st.markdown("**Static population reference evidence (NB5 / NB6)**")
    status_display = {
        "POPULATION_EVALUATION_AVAILABLE": st.success,
        "INSUFFICIENT_GROUP_DATA": st.warning,
        "FAIRNESS_EVALUATION_BLOCKED": st.error,
    }.get(static["evaluation_status"], st.info)
    status_display(f"`{static['evaluation_status']}` — {static['reason']}")

    if r["population_data_available"]:
        st.write(
            f"Held-out test cohort: **{static['test_cohort_rows']}** rows, "
            f"**{static['test_cohort_unique_participants']}** unique participants."
        )

        st.markdown("*Group sample sizes*")
        st.dataframe(
            [
                {
                    "group_variable": g["group_variable"],
                    "group_label": g["group_label"],
                    "n": g["n"],
                    "positives": g["positives"],
                    "insufficient_data": g["insufficient_data"],
                }
                for g in static["group_definitions"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        insufficient = [g for g in static["group_definitions"] if g["insufficient_data"]]
        for g in insufficient:
            st.warning(
                f"**{g['group_variable']} = {g['group_label']}**: "
                f"{g['insufficient_data_note']}"
            )

        with st.expander("Full population-level metrics by dimension (NB5 / NB6)"):
            for dimension, rows in static["metrics"].items():
                st.markdown(f"**{dimension}**")
                st.dataframe(
                    [
                        {
                            "indicator": row["indicator"],
                            "value": row["value"],
                            "unit": row["unit"],
                            "risk_level": row["risk_level"],
                        }
                        for row in rows
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        for err in static.get("load_errors", []):
            st.error(err)

    with st.expander("Data limitations"):
        for item in r["data_limitations"]:
            st.write(f"- {item}")

    st.caption(
        f"Clinical interpretation allowed (inherited, unchanged from "
        f"upstream agents): `{r['clinical_interpretation_allowed']}`"
    )

    with st.expander("Full structured agent output (JSON contract)"):
        st.json(r)


def render_safety_result(result: SafetyResult) -> None:
    r = result.as_dict

    decision_display = {
        "BLOCK": st.error,
        "TECHNICAL_ONLY": st.warning,
        "CLINICAL_INTERPRETATION_GATE_PASSED": st.success,
    }.get(r["safety_decision"], st.info)
    decision_display(f"**Safety decision:** `{r['safety_decision']}`")

    if r["clinical_gate_disclaimer"]:
        st.caption(r["clinical_gate_disclaimer"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Technical prediction allowed", str(r["technical_prediction_allowed"]))
    col2.metric("Technical explanation allowed", str(r["technical_explanation_allowed"]))
    col3.metric("Clinical interpretation allowed", str(r["clinical_interpretation_allowed"]))

    st.markdown("**Reason codes**")
    for code in r["reason_codes"]:
        st.write(f"- `{code}`")

    st.markdown("**Observed vs. missing input summary**")
    summary = r["input_observation_summary"]
    st.write(
        f"Raw features observed: **{summary['observed_count']} / "
        f"{summary['total_raw_features']}** "
        f"(missing: {summary['missing_count']})"
    )
    if summary["contribution_audit_note"]:
        st.warning(summary["contribution_audit_note"])
    if summary["contribution_observation_audit"]:
        with st.expander(
            f"Top-Contribution Audit — NOT exhaustive "
            f"(top {summary['contribution_audit_limit']} of "
            f"{summary['total_transformed_features']} transformed features)"
        ):
            st.dataframe(
                [
                    {
                        "feature": c["feature_label"],
                        "raw_input_status": c["raw_input_status"],
                        "contribution_value_status": c["contribution_value_status"],
                        "contribution": c["contribution"],
                    }
                    for c in summary["contribution_observation_audit"]
                ],
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("**Population evidence status**")
    pop = r["population_evidence_status"]
    st.write(
        f"This session: `{pop['current_session_status']}`  ·  "
        f"Static reference: `{pop['static_reference_status']}`"
    )
    for b in pop["boundaries"]:
        st.caption(f"- {b}")

    st.markdown("**Explanation interpretation boundaries**")
    for b in r["explanation_interpretation_status"]["boundaries"]:
        st.caption(f"- {b}")

    with st.expander("Full structured agent output (JSON contract)"):
        st.json(r)


def render_agent_pipeline_scaffold(
    governance: Governance,
    input_quality_result: InputQualityResult | None,
    prediction_result: PredictionResult | None,
    explainability_result: ExplainabilityResult | None,
    trust_fairness_result: TrustFairnessResult | None,
    safety_result: SafetyResult | None,
) -> None:
    st.subheader("Seven-Agent Workflow")
    st.caption(
        "Fixed execution order. The order and presence of every stage is "
        "fixed and will not change across phases."
    )

    phase_map = {
        "Input/Data Quality Agent": "Phase 2",
        "Prediction Agent": "Phase 3",
        "Explainability Agent": "Phase 4",
        "Trust & Fairness Agent": "Phase 5",
        "Safety Agent": "Phase 6 — mandatory control point",
        "CDS/Reporting Agent": "Phase 7 (not yet implemented)",
        "Agentic Orchestrator": "Phase 8 (not yet implemented)",
    }

    for agent_name in governance.agent_execution_order:
        is_input_quality = agent_name == "Input/Data Quality Agent"
        is_prediction = agent_name == "Prediction Agent"
        is_explainability = agent_name == "Explainability Agent"
        is_trust_fairness = agent_name == "Trust & Fairness Agent"
        is_safety = agent_name == "Safety Agent"
        has_result = (
            (is_input_quality and input_quality_result is not None)
            or (is_prediction and prediction_result is not None)
            or (is_explainability and explainability_result is not None)
            or (is_trust_fairness and trust_fairness_result is not None)
            or (is_safety and safety_result is not None)
        )
        icon = "✅" if has_result else "🔲"

        with st.expander(
            f"{icon} {agent_name} — {phase_map.get(agent_name, 'unscheduled')}",
            expanded=has_result,
        ):
            if is_input_quality:
                if input_quality_result is not None:
                    render_input_quality_result(input_quality_result)
                else:
                    st.write(
                        "Not yet executed. Submit the patient input form "
                        "above to run this agent."
                    )
            elif is_prediction:
                if prediction_result is not None:
                    render_prediction_result(prediction_result, safety_result)
                else:
                    st.write(
                        "Not yet executed. Submit the patient input form "
                        "above to run the Input/Data Quality Agent, which "
                        "this agent runs immediately after."
                    )
            elif is_explainability:
                if explainability_result is not None:
                    render_explainability_result(explainability_result, safety_result)
                else:
                    st.write(
                        "Not yet executed. Submit the patient input form "
                        "above to run the earlier agents, which this "
                        "agent runs immediately after."
                    )
            elif is_trust_fairness:
                if trust_fairness_result is not None:
                    render_trust_fairness_result(trust_fairness_result)
                else:
                    st.write(
                        "Not yet executed. Submit the patient input form "
                        "above to run the earlier agents, which this "
                        "agent runs immediately after."
                    )
            elif is_safety:
                if safety_result is not None:
                    render_safety_result(safety_result)
                else:
                    st.write(
                        "Not yet executed. Submit the patient input form "
                        "above to run the earlier agents, which this "
                        "agent runs immediately after."
                    )
            else:
                st.write("Not yet implemented in this phase.")

            if agent_name == governance.mandatory_safety_control_point:
                st.info(
                    "This is the mandatory safety control point. Downstream "
                    "agents (once implemented) are architecturally unable to "
                    "present a clinical interpretation when this agent's "
                    "safety decision blocks it — this is enforced above by "
                    "gating the Prediction and Explainability sections on "
                    "this agent's own flags, not only on their own internal "
                    "ones."
                )


def render_governance_notes(governance: Governance) -> None:
    with st.expander("Governance notes"):
        for note in governance.governance_notes:
            st.write(f"- {note}")


def main() -> None:
    try:
        governance = get_governance()
    except GovernanceConfigError as exc:
        st.set_page_config  # no-op, page config already set above
        st.error("Governance configuration failed to load. The application cannot start.")
        st.exception(exc)
        st.stop()
        return

    render_banner(governance.research_only_banner)

    st.title(governance.project_name)
    st.write(
        "This is a research demonstrator that operationalizes a validated, "
        "locked, seven-agent clinical-AI workflow. It does not diagnose, "
        "treat, or produce a calibrated individual clinical risk estimate."
    )

    bundle = get_model_bundle(governance)

    render_integrity_panel(governance, bundle)

    if not bundle.is_safe_to_use:
        st.error(
            "The application cannot proceed: the locked model/preprocessor "
            "failed integrity verification or could not be safely loaded. "
            "No downstream agent will run. This is expected fail-safe "
            "behavior, not a bug."
        )
        st.stop()
        return

    try:
        schema = get_schema()
    except SchemaError as exc:
        st.error(
            "The locked 214-feature input schema failed to load or "
            "failed internal consistency validation. The application "
            "cannot proceed."
        )
        st.exception(exc)
        st.stop()
        return

    st.sidebar.header("About this demonstrator")
    st.sidebar.info(
        "Fill in the patient input form below with whatever information "
        "is known. Everything is optional — leave fields blank if unknown. "
        "Blank fields are recorded as missing and will correctly limit or "
        "block clinical interpretation downstream; nothing is ever "
        "invented to fill a gap."
    )
    st.sidebar.write(f"Locked model input features: **{len(schema.locked_features)}**")
    st.sidebar.write(f"Prediction-time clinical measurements: **{len(schema.clinical_measurements)}**")

    submitted, patient_record = render_patient_input_form(schema)

    if submitted and patient_record is not None:
        input_quality_result = run_input_quality_agent(patient_record, schema)
        st.session_state["input_quality_result"] = input_quality_result
        st.session_state["patient_record"] = patient_record

        prediction_result = run_prediction_agent(
            patient_record, schema, bundle, governance, input_quality_result
        )
        st.session_state["prediction_result"] = prediction_result

        explainability_result = run_explainability_agent(
            patient_record, schema, bundle, governance, input_quality_result, prediction_result
        )
        st.session_state["explainability_result"] = explainability_result

        trust_fairness_result = run_trust_fairness_agent(input_quality_result, prediction_result)
        st.session_state["trust_fairness_result"] = trust_fairness_result

        safety_result = run_safety_agent(
            patient_record,
            schema,
            input_quality_result,
            prediction_result,
            explainability_result,
            trust_fairness_result,
        )
        st.session_state["safety_result"] = safety_result

    input_quality_result = st.session_state.get("input_quality_result")
    prediction_result = st.session_state.get("prediction_result")
    explainability_result = st.session_state.get("explainability_result")
    trust_fairness_result = st.session_state.get("trust_fairness_result")
    safety_result = st.session_state.get("safety_result")

    st.divider()
    render_agent_pipeline_scaffold(
        governance,
        input_quality_result,
        prediction_result,
        explainability_result,
        trust_fairness_result,
        safety_result,
    )
    render_governance_notes(governance)


if __name__ == "__main__":
    main()

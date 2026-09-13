"""
Trustworthy Multi-Agent Healthcare AI Demonstrator — Streamlit app shell.

PHASE 1 SCOPE ONLY:
  - Persistent research-only banner
  - Startup governance load + locked-artifact integrity verification
  - A visible, fixed-order scaffold for the seven agents (placeholders)

No prediction, explanation, fairness, safety, or reporting logic exists yet.
Those are added in later phases and will plug into the placeholder sections
below without changing this file's overall structure.
"""

from __future__ import annotations

import streamlit as st

from backend.governance import Governance, GovernanceConfigError, load_governance
from backend.model_loader import ModelBundle, load_model_bundle
from backend.schema import LockedSchema, SchemaError, load_locked_schema
from backend.agents.input_quality_agent import InputQualityResult, run_input_quality_agent
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


def render_agent_pipeline_scaffold(
    governance: Governance,
    input_quality_result: InputQualityResult | None,
) -> None:
    st.subheader("Seven-Agent Workflow")
    st.caption(
        "Fixed execution order. The order and presence of every stage is "
        "fixed and will not change across phases."
    )

    phase_map = {
        "Input/Data Quality Agent": "Phase 2",
        "Prediction Agent": "Phase 3 (not yet implemented)",
        "Explainability Agent": "Phase 4 (not yet implemented)",
        "Trust & Fairness Agent": "Phase 5 (not yet implemented)",
        "Safety Agent": "Phase 6 (not yet implemented) — mandatory control point",
        "CDS/Reporting Agent": "Phase 7 (not yet implemented)",
        "Agentic Orchestrator": "Phase 8 (not yet implemented)",
    }

    for agent_name in governance.agent_execution_order:
        is_input_quality = agent_name == "Input/Data Quality Agent"
        has_result = is_input_quality and input_quality_result is not None
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
            else:
                st.write("Not yet implemented in this phase.")

            if agent_name == governance.mandatory_safety_control_point:
                st.info(
                    "This is the mandatory safety control point. Once "
                    "implemented, downstream agents will be architecturally "
                    "unable to present a clinical interpretation when this "
                    "agent's safety gate is blocked."
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

    input_quality_result = st.session_state.get("input_quality_result")

    st.divider()
    render_agent_pipeline_scaffold(governance, input_quality_result)
    render_governance_notes(governance)


if __name__ == "__main__":
    main()

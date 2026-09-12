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


def render_agent_pipeline_scaffold(governance: Governance) -> None:
    st.subheader("Seven-Agent Workflow")
    st.caption(
        "Fixed execution order. Each stage below will be implemented in "
        "later phases; the order and presence of every stage is fixed now "
        "and will not change."
    )

    phase_map = {
        "Input/Data Quality Agent": "Phase 2 (not yet implemented)",
        "Prediction Agent": "Phase 3 (not yet implemented)",
        "Explainability Agent": "Phase 4 (not yet implemented)",
        "Trust & Fairness Agent": "Phase 5 (not yet implemented)",
        "Safety Agent": "Phase 6 (not yet implemented) — mandatory control point",
        "CDS/Reporting Agent": "Phase 7 (not yet implemented)",
        "Agentic Orchestrator": "Phase 8 (not yet implemented)",
    }

    for agent_name in governance.agent_execution_order:
        with st.expander(f"🔲 {agent_name} — {phase_map.get(agent_name, 'unscheduled')}"):
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

    st.sidebar.header("Patient Input")
    st.sidebar.info(
        "A structured patient-input form will be added in Phase 2 "
        "(Input/Data Quality Agent). It will map to the locked 214-feature "
        "NB3 schema without requiring manual entry of all 214 features, and "
        "will never fabricate or silently impute missing values."
    )

    render_agent_pipeline_scaffold(governance)
    render_governance_notes(governance)


if __name__ == "__main__":
    main()

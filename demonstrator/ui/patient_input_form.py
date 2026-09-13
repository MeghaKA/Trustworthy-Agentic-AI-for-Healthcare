"""
Structured patient-input form.

Renders ONLY the curated, practical subset of the locked 214-feature
schema defined in backend/schema.py as UI_EXPOSED_FIELD_GROUPS (30
features). Every one of the other 184 locked features is never rendered
anywhere in this form — not in a collapsed section, not optionally — it
is simply absent from the UI and is always set to None in the
constructed patient record.

Every rendered field defaults to None/empty — nothing is pre-filled with
a placeholder numeric value that could be mistaken for an observed
patient measurement. A field only becomes "observed" if the person using
the demonstrator actively enters something.

This module always produces a record with a key for every one of the 214
locked features (value None for anything not exposed in the UI or left
blank) plus the optional SEQN identifier, matching the shape the
Input/Data Quality Agent expects.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from backend.schema import LockedSchema, IDENTIFIER_COLUMN

# Plain-English glosses for the 30 exposed fields only. These describe
# what each standard NHANES variable code represents; they do not assign
# or imply any clinical value or interpretation.
EXPOSED_FIELD_LABELS: dict[str, str] = {
    "RIAGENDR": "Gender (NHANES code: 1 = Male, 2 = Female)",
    "RIDAGEYR": "Age (years)",
    "RIDRETH3": "Race/ethnicity — NHANES code",
    "DMDEDUC2": "Education level (NHANES code)",
    "DMDMARTZ": "Marital status (NHANES code)",
    "INDFMPIR": "Ratio of family income to poverty threshold",
    "BMXWT": "Weight (kg)",
    "BMXHT": "Standing height (cm)",
    "BMXBMI": "Body mass index (kg/m²)",
    "BMXWAIST": "Waist circumference (cm)",
    "BMXHIP": "Hip circumference (cm)",
    "BPXOSY1": "Systolic blood pressure (mmHg)",
    "BPXODI1": "Diastolic blood pressure (mmHg)",
    "BPXOPLS1": "Pulse rate (bpm)",
    "LBXGLU": "Fasting plasma glucose (mg/dL)",
    "LBDGLUSI": "Fasting plasma glucose (mmol/L, SI units)",
    "LBXGH": "Glycohemoglobin / HbA1c (%)",
    "URXUMA": "Urine albumin (µg/mL)",
    "URXUMS": "Urine albumin (mg/L)",
    "URXUCR": "Urine creatinine (mg/dL)",
    "URXCRS": "Urine creatinine (µmol/L, SI units)",
    "LBXTC": "Total cholesterol (mg/dL)",
    "LBDHDD": "HDL cholesterol (mg/dL)",
    "LBDLDL": "LDL cholesterol (mg/dL)",
    "LBXTR": "Triglycerides (mg/dL)",
    "HUQ010": "General health condition, self-reported (NHANES code)",
    "HUQ030": "Has a routine place for healthcare (NHANES code)",
    "HUQ051": "Healthcare visits in past year (NHANES code)",
    "HUQ071": "Overnight hospital stay in past year (NHANES code)",
    "HUQ090": "Seen a mental health professional in past year (NHANES code)",
}

CLINICAL_MEASUREMENT_LABEL_NOTE = (
    "These seven fields are the prediction-time clinical measurements "
    "registered in the validated pipeline. If they are left blank, "
    "downstream clinical interpretation will be blocked by design — "
    "this is expected safety behavior, not an error."
)


def _numeric_field(feature: str, key_prefix: str) -> Any:
    label = EXPOSED_FIELD_LABELS.get(feature, feature)
    return st.number_input(
        label=f"{label}  ·  `{feature}`",
        value=None,
        format="%.4f",
        key=f"{key_prefix}_{feature}",
        help="Leave blank if unknown. Blank is recorded as missing, never as zero.",
    )


def render_patient_input_form(schema: LockedSchema) -> tuple[bool, dict[str, Any] | None]:
    """
    Render the structured patient-input form.

    Only the curated subset in schema.ui_exposed_field_groups is rendered.
    All other locked features are set to None without any UI element.

    Returns (submitted, patient_record). patient_record is None unless the
    form was just submitted; when submitted, it contains a key for every
    one of the 214 locked features (None where not exposed or left blank)
    plus SEQN.
    """
    st.subheader("Patient Input")
    st.caption(
        "All fields below are optional. Leave anything unknown blank — "
        "it will be recorded as missing, never guessed or filled in. This "
        "form intentionally shows only a small, practical subset of the "
        "214 fields the locked model actually uses; fields not shown here "
        "are simply not collected and are treated as missing. Incomplete "
        "or missing information will limit — and, for the clinical "
        "measurements below, will block — downstream clinical "
        "interpretation. That is expected, correct safety behavior, not "
        "a limitation of this form."
    )

    exposed_values: dict[str, Any] = {}

    with st.form("patient_input_form", clear_on_submit=False):
        seqn_input = st.text_input(
            "Patient identifier (optional — for provenance/display only; "
            "never passed to the model)",
            value="",
            key="patient_seqn",
        )

        for group_key, group in schema.ui_exposed_field_groups.items():
            st.markdown(f"**{group['label']}**")
            if group_key == "clinical_measurements_prediction_time":
                st.caption(CLINICAL_MEASUREMENT_LABEL_NOTE)
            cols = st.columns(2)
            for i, feature in enumerate(group["features"]):
                with cols[i % 2]:
                    exposed_values[feature] = _numeric_field(feature, "core")

        submitted = st.form_submit_button("Run Input / Data Quality Assessment")

    if not submitted:
        return False, None

    # Every one of the 214 locked features gets a key. Anything not in the
    # exposed subset is explicitly None — never rendered, never fabricated.
    patient_record: dict[str, Any] = {feature: None for feature in schema.locked_features}
    patient_record.update(exposed_values)
    patient_record[IDENTIFIER_COLUMN] = seqn_input.strip() if seqn_input.strip() else None

    return True, patient_record

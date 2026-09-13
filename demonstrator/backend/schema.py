"""
Locked input schema module.

Loads the authoritative 214-feature schema for the locked NB3 preprocessor
from artifacts/schema/nb3_locked_214_feature_schema.csv (copied verbatim
from the NB7 export, which itself was derived directly from the locked
preprocessor's `feature_names_in_` — see NB9 Cell 13). This module does
not invent, reorder, add, or remove any feature.

This module additionally defines two field groupings over those exact 214
features:

  - FIELD_GROUPS: a complete partition of all 214 features into named
    groups, used only to validate that nothing is lost, duplicated, or
    invented. It is NOT used to render the UI, and most of its groups are
    never shown to the user.

  - UI_EXPOSED_FIELD_GROUPS: a small, curated subset of FIELD_GROUPS
    (30 of 214 features) that IS what the normal user-facing form renders.
    Every feature not listed here is simply never part of the UI at all —
    not hidden in a collapsed "advanced" section, just absent from the
    form — and is always recorded as missing/None in the constructed
    214-feature backend record.

Neither grouping assigns, infers, or implies any clinical value for any
feature; they exist purely to organize known, standard NHANES codebook
variable families for usability. Both are validated at import time: the
former as an exact partition of the 214 features, the latter as a subset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = PROJECT_ROOT / "artifacts" / "schema" / "nb3_locked_214_feature_schema.csv"

IDENTIFIER_COLUMN = "SEQN"
TARGET_COLUMN = "diabetes_target"

# The seven prediction-time clinical measurements. Confirmed identically in
# NB7 (nb7_prototype_clinical_measurement_audit.csv), NB8 Cell 15, and NB9
# Cells 13/14.
CLINICAL_MEASUREMENTS: tuple[str, ...] = (
    "LBXGLU",
    "LBDGLUSI",
    "LBXGH",
    "URXUMA",
    "URXUMS",
    "URXUCR",
    "URXCRS",
)

# -----------------------------------------------------------------------
# FULL 214-feature grouping (used for schema completeness validation only
# — NOT used to render the UI). Every one of the 214 locked features
# appears here exactly once, so we can prove nothing is lost or invented
# even though most of these groups are never shown to the user.
# -----------------------------------------------------------------------
FIELD_GROUPS: dict[str, dict[str, object]] = {
    "core_demographics": {
        "label": "Demographics",
        "features": [
            "RIAGENDR", "RIDAGEYR", "RIDRETH1", "RIDRETH3",
            "DMDEDUC2", "DMDMARTZ", "INDFMPIR",
        ],
    },
    "core_body_measurements": {
        "label": "Body Measurements",
        "features": [
            "BMXWT", "BMXHT", "BMXBMI", "BMXLEG",
            "BMXARML", "BMXARMC", "BMXWAIST", "BMXHIP",
        ],
    },
    "core_blood_pressure": {
        "label": "Blood Pressure",
        "features": [
            "BPAOARM", "BPAOCSZ",
            "BPXOSY1", "BPXODI1", "BPXOSY2", "BPXODI2", "BPXOSY3", "BPXODI3",
            "BPXOPLS1", "BPXOPLS2", "BPXOPLS3",
        ],
    },
    "clinical_measurements_prediction_time": {
        "label": "Prediction-Time Clinical Measurements",
        "features": list(CLINICAL_MEASUREMENTS),
    },
    "core_lipid_panel": {
        "label": "Lipid Panel",
        "features": [
            "LBXTC", "LBDTCSI", "LBXTR", "LBDTRSI",
            "LBDLDL", "LBDLDLSI", "LBDLDLM", "LBDLDMSI", "LBDLDLN", "LBDLDNSI",
            "LBDHDD", "LBDHDDSI",
        ],
    },
    "core_health_status": {
        "label": "General Health Status & Care Access",
        "features": ["HUQ010", "HUQ030", "HUQ051", "HUQ071", "HUQ090"],
    },
    "cbc_panel": {
        "label": "Complete Blood Count",
        "features": [
            "LBXWBCSI", "LBXLYPCT", "LBXMOPCT", "LBXNEPCT", "LBXEOPCT", "LBXBAPCT",
            "LBDLYMNO", "LBDMONO", "LBDNENO", "LBDEONO", "LBDBANO",
            "LBXRBCSI", "LBXHGB", "LBXHCT", "LBXMCVSI", "LBXMC", "LBXMCHSI",
            "LBXRDW", "LBXPLTSI", "LBXMPSI", "LBXNRBC",
        ],
    },
    "inflammation": {
        "label": "Inflammation Markers",
        "features": ["LBXHSCRP", "LBDHRPLC"],
    },
    "metabolic_panel": {
        "label": "Comprehensive Metabolic Panel",
        "features": [
            "LBXSATSI", "LBDSATLC", "LBXSAL", "LBDSALSI", "LBXSAPSI", "LBXSASSI",
            "LBXSC3SI", "LBXSBU", "LBDSBUSI", "LBXSCLSI", "LBXSCK", "LBXSCR",
            "LBDSCRSI", "LBXSGB", "LBDSGBSI", "LBXSGL", "LBDSGLSI", "LBXSGTSI",
            "LBDSGTLC", "LBXSIR", "LBDSIRSI", "LBXSLDSI", "LBXSOSSI", "LBXSPH",
            "LBDSPHSI", "LBXSKSI", "LBXSNASI", "LBXSTB", "LBDSTBSI", "LBDSTBLC",
            "LBXSCA", "LBDSCASI", "LBXSCH", "LBDSCHSI", "LBXSTP", "LBDSTPSI",
            "LBXSTR", "LBDSTRSI", "LBXSUA", "LBDSUASI",
        ],
    },
    "urine_kidney": {
        "label": "Urine / Kidney Derived Values",
        "features": ["URDUMALC", "URDUCRLC", "URDACT"],
    },
    "diet_behavior": {
        "label": "Diet & Nutrition Behavior",
        "features": [
            "DBQ700", "DBQ197", "DBQ223A", "DBQ223B", "DBQ229",
            "DBQ235A", "DBQ235B", "DBQ235C", "DBQ301", "DBQ330",
            "DBQ360", "DBQ370", "DBD381", "DBQ390", "DBQ400", "DBD411",
            "DBD895", "DBD900", "DBD905", "DBD910",
            "CBQ596", "CBQ606", "CBQ611", "DBQ930", "DBQ935", "DBQ940", "DBQ945",
        ],
    },
    "physical_activity": {
        "label": "Physical Activity",
        "features": [
            "PAQ605", "PAQ610", "PAD615", "PAQ620", "PAQ625", "PAD630",
            "PAQ635", "PAQ640", "PAD645", "PAQ650", "PAQ655", "PAD660",
            "PAQ665", "PAQ670", "PAD675", "PAD680",
        ],
    },
    "sleep": {
        "label": "Sleep",
        "features": [
            "SLQ300", "SLQ310", "SLD012", "SLQ320", "SLQ330", "SLD013",
            "SLQ030", "SLQ040", "SLQ050", "SLQ120",
        ],
    },
    "alcohol": {
        "label": "Alcohol Use",
        "features": [
            "ALQ111", "ALQ121", "ALQ130", "ALQ142",
            "ALQ270", "ALQ280", "ALQ151", "ALQ170",
        ],
    },
    "smoking": {
        "label": "Smoking / Tobacco Use",
        "features": [
            "SMQ020", "SMD030", "SMQ040", "SMQ050Q", "SMQ050U", "SMD057",
            "SMQ078", "SMD641", "SMD650", "SMD100FL", "SMD100MN", "SMQ670",
        ],
    },
    "health_insurance": {
        "label": "Health Insurance",
        "features": ["HIQ011", "HIQ032A", "HIQ032B", "HIQ032D", "HIQ105", "HIQ270", "HIQ210"],
    },
    "administrative_survey_methodology": {
        "label": "NHANES Survey / Interview Administrative Fields",
        "features": [
            "SDDSRVYR", "RIDSTATR", "RIDEXMON", "DMDBORN4", "DMDYRUSZ",
            "SIALANG", "SIAPROXY", "SIAINTRP", "FIALANG", "FIAPROXY", "FIAINTRP",
            "MIALANG", "MIAPROXY", "MIAINTRP", "AIALANGA", "BMDSTATS", "BMDBMIC",
            "SMAQUEX2",
        ],
    },
}


# -----------------------------------------------------------------------
# PRACTICAL UI SUBSET — the ONLY fields the normal user-facing form
# renders. This is a small, curated selection (30 of 214), chosen to be
# things a person could plausibly know or have on hand, with redundant
# unit/method duplicates and equipment-metadata fields left out. Every
# feature not listed here is never shown in the UI and is always recorded
# as None/missing — it is NOT hidden in a collapsed "advanced" section,
# it simply is not part of the form at all.
#
# The seven prediction-time clinical measurements are always included in
# full, unabridged, since they directly gate the safety decision.
# -----------------------------------------------------------------------
UI_EXPOSED_FIELD_GROUPS: dict[str, dict[str, object]] = {
    "demographics": {
        "label": "Demographics",
        "features": ["RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "DMDMARTZ", "INDFMPIR"],
    },
    "body_measurements": {
        "label": "Body Measurements",
        "features": ["BMXWT", "BMXHT", "BMXBMI", "BMXWAIST", "BMXHIP"],
    },
    "blood_pressure": {
        "label": "Blood Pressure",
        "features": ["BPXOSY1", "BPXODI1", "BPXOPLS1"],
    },
    "clinical_measurements_prediction_time": {
        "label": "Prediction-Time Clinical Measurements",
        "features": list(CLINICAL_MEASUREMENTS),
    },
    "lipid_panel": {
        "label": "Lipid Panel",
        "features": ["LBXTC", "LBDHDD", "LBDLDL", "LBXTR"],
    },
    "health_status": {
        "label": "General Health Status & Care Access",
        "features": ["HUQ010", "HUQ030", "HUQ051", "HUQ071", "HUQ090"],
    },
}


class SchemaError(RuntimeError):
    """Raised when the locked schema file is missing, malformed, or inconsistent
    with FIELD_GROUPS."""


@dataclass(frozen=True)
class LockedSchema:
    locked_features: tuple[str, ...]  # exact order from the authoritative CSV
    feature_info: dict[str, dict[str, object]]  # feature -> {dtype, transformer_role, missing_fraction}
    numeric_features: frozenset[str]
    categorical_features: frozenset[str]
    clinical_measurements: tuple[str, ...]
    field_groups: dict[str, dict[str, object]]
    ui_exposed_field_groups: dict[str, dict[str, object]]


def load_locked_schema() -> LockedSchema:
    if not SCHEMA_PATH.exists():
        raise SchemaError(f"Locked schema file not found at: {SCHEMA_PATH}")

    df = pd.read_csv(SCHEMA_PATH)
    required_columns = {"feature", "dtype", "transformer_role", "missing_fraction"}
    if not required_columns.issubset(df.columns):
        raise SchemaError(
            f"Locked schema file at {SCHEMA_PATH} is missing required "
            f"columns. Found: {list(df.columns)}"
        )

    locked_features = tuple(df["feature"].astype(str).tolist())

    if len(locked_features) != 214:
        raise SchemaError(
            f"Locked schema at {SCHEMA_PATH} has {len(locked_features)} "
            "features; expected exactly 214. Refusing to proceed with an "
            "unexpected schema."
        )

    feature_info: dict[str, dict[str, object]] = {}
    numeric_features: set[str] = set()
    categorical_features: set[str] = set()

    for _, row in df.iterrows():
        feature = str(row["feature"])
        role = str(row["transformer_role"])
        feature_info[feature] = {
            "dtype": str(row["dtype"]),
            "transformer_role": role,
            "missing_fraction": float(row["missing_fraction"]),
        }
        if role == "numeric":
            numeric_features.add(feature)
        elif role == "categorical":
            categorical_features.add(feature)
        else:
            raise SchemaError(
                f"Feature {feature!r} has unrecognized transformer_role "
                f"{role!r}; expected 'numeric' or 'categorical'."
            )

    # Validate FIELD_GROUPS is an exact partition of locked_features.
    grouped_features: list[str] = []
    for group in FIELD_GROUPS.values():
        grouped_features.extend(group["features"])  # type: ignore[arg-type]

    grouped_set = set(grouped_features)
    locked_set = set(locked_features)

    if len(grouped_features) != len(grouped_set):
        raise SchemaError("FIELD_GROUPS contains duplicate feature assignments.")
    if grouped_set != locked_set:
        missing = locked_set - grouped_set
        extra = grouped_set - locked_set
        raise SchemaError(
            "FIELD_GROUPS does not exactly match the locked 214-feature "
            f"schema. Missing from groups: {sorted(missing)}. "
            f"Not part of locked schema: {sorted(extra)}."
        )

    if not set(CLINICAL_MEASUREMENTS).issubset(locked_set):
        raise SchemaError(
            "One or more registered clinical measurements are not present "
            "in the locked 214-feature schema."
        )

    # Validate UI_EXPOSED_FIELD_GROUPS is a subset (not a partition) of the
    # locked 214-feature schema, with no duplicate assignments, and that it
    # contains the full set of clinical measurements unabridged.
    exposed_features: list[str] = []
    for group in UI_EXPOSED_FIELD_GROUPS.values():
        exposed_features.extend(group["features"])  # type: ignore[arg-type]

    exposed_set = set(exposed_features)

    if len(exposed_features) != len(exposed_set):
        raise SchemaError("UI_EXPOSED_FIELD_GROUPS contains duplicate feature assignments.")
    if not exposed_set.issubset(locked_set):
        raise SchemaError(
            "UI_EXPOSED_FIELD_GROUPS references features not present in "
            f"the locked 214-feature schema: {sorted(exposed_set - locked_set)}"
        )
    if not set(CLINICAL_MEASUREMENTS).issubset(exposed_set):
        raise SchemaError(
            "UI_EXPOSED_FIELD_GROUPS must include all seven prediction-time "
            "clinical measurements unabridged; one or more is missing."
        )

    return LockedSchema(
        locked_features=locked_features,
        feature_info=feature_info,
        numeric_features=frozenset(numeric_features),
        categorical_features=frozenset(categorical_features),
        clinical_measurements=CLINICAL_MEASUREMENTS,
        field_groups=FIELD_GROUPS,
        ui_exposed_field_groups=UI_EXPOSED_FIELD_GROUPS,
    )

"""
Governance module.

This module defines the SINGLE authoritative source for every governance
fact the application needs at runtime:

  - the locked decision threshold (sourced ONLY from NB3's
    early_detection_model_metadata.json — never duplicated elsewhere)
  - the authoritative sha256 hashes for the locked model/preprocessor
    (sourced from config/governance_config.json, cross-checked against
    NB8/NB9 provenance during development)
  - the fixed seven-agent execution order
  - the persistent research-only banner text

No other module should hardcode the threshold, the hashes, or the agent
order. Everything else in the application should import a Governance
instance from here.

This module does not load the model itself (see backend/model_loader.py)
and does not perform any prediction, explanation, or clinical logic. It is
pure configuration plumbing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Project root = two levels up from this file (backend/governance.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOVERNANCE_CONFIG_PATH = PROJECT_ROOT / "config" / "governance_config.json"


class GovernanceConfigError(RuntimeError):
    """Raised when governance configuration cannot be loaded or is malformed.

    This is intentionally a loud failure. If governance configuration is
    missing or inconsistent, the application must not silently continue.
    """


@dataclass(frozen=True)
class ArtifactRecord:
    """Authoritative record for a single locked artifact."""

    description: str
    path: Path
    sha256: str | None = None


@dataclass(frozen=True)
class Governance:
    """
    Immutable, fully-resolved governance facts for this application run.

    Attributes mirror the two source files (governance_config.json and the
    NB3 model metadata) but are exposed as a single typed object so the rest
    of the codebase never has to re-parse JSON or guess a key name.
    """

    project_name: str
    research_only_banner: str

    model_artifact: ArtifactRecord
    preprocessor_artifact: ArtifactRecord
    metadata_path: Path

    required_sklearn_version: str
    fail_safe_on_mismatch: bool

    agent_execution_order: tuple[str, ...]
    mandatory_safety_control_point: str
    governance_notes: tuple[str, ...]

    # Sourced exclusively from NB3 metadata — the single source of truth.
    locked_threshold: float
    threshold_selection_method: str
    target_sensitivity: float
    raw_feature_count: int
    processed_feature_count: int

    raw_metadata: dict[str, Any] = field(repr=False)
    raw_config: dict[str, Any] = field(repr=False)


def _load_json(path: Path, *, what: str) -> dict[str, Any]:
    if not path.exists():
        raise GovernanceConfigError(
            f"Required {what} file not found at: {path}. "
            "The application cannot start without it."
        )
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise GovernanceConfigError(
            f"Required {what} file at {path} is not valid JSON: {exc}"
        ) from exc


def load_governance() -> Governance:
    """
    Load and validate governance configuration + NB3 metadata.

    Raises GovernanceConfigError if either source file is missing, malformed,
    or missing a required field. This function performs NO hashing and NO
    model loading — it only assembles configuration facts. Hash
    verification and model loading happen in backend/model_loader.py, which
    consumes the Governance object this function returns.
    """
    config = _load_json(GOVERNANCE_CONFIG_PATH, what="governance_config.json")

    try:
        artifacts = config["authoritative_artifacts"]
        model_cfg = artifacts["tier_1_locked_model"]
        preproc_cfg = artifacts["tier_1_locked_preprocessor"]
        metadata_cfg = artifacts["tier_1_model_metadata"]
        hash_cfg = config["hash_verification"]
    except KeyError as exc:
        raise GovernanceConfigError(
            f"governance_config.json is missing required key: {exc}"
        ) from exc

    metadata_path = PROJECT_ROOT / metadata_cfg["path"]
    metadata = _load_json(metadata_path, what="NB3 model metadata")

    try:
        threshold_info = metadata["threshold_information"]
        locked_threshold = float(threshold_info["threshold"])
        threshold_selection_method = str(threshold_info["threshold_selection_method"])
        target_sensitivity = float(threshold_info["target_sensitivity"])
        raw_feature_count = int(metadata["raw_features"])
        processed_feature_count = int(metadata["processed_features"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GovernanceConfigError(
            f"NB3 model metadata at {metadata_path} is missing or has an "
            f"invalid required field: {exc}"
        ) from exc

    model_artifact = ArtifactRecord(
        description=model_cfg["description"],
        path=PROJECT_ROOT / model_cfg["path"],
        sha256=model_cfg["sha256"],
    )
    preprocessor_artifact = ArtifactRecord(
        description=preproc_cfg["description"],
        path=PROJECT_ROOT / preproc_cfg["path"],
        sha256=preproc_cfg["sha256"],
    )

    return Governance(
        project_name=config["project_name"],
        research_only_banner=config["research_only_banner"],
        model_artifact=model_artifact,
        preprocessor_artifact=preprocessor_artifact,
        metadata_path=metadata_path,
        required_sklearn_version=hash_cfg["required_sklearn_version"],
        fail_safe_on_mismatch=bool(hash_cfg["fail_safe_on_mismatch"]),
        agent_execution_order=tuple(config["agent_execution_order"]),
        mandatory_safety_control_point=config["mandatory_safety_control_point"],
        governance_notes=tuple(config["governance_notes"]),
        locked_threshold=locked_threshold,
        threshold_selection_method=threshold_selection_method,
        target_sensitivity=target_sensitivity,
        raw_feature_count=raw_feature_count,
        processed_feature_count=processed_feature_count,
        raw_metadata=metadata,
        raw_config=config,
    )

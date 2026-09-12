"""
Model loading module.

Responsibility: verify the integrity of the two locked NB3 artifacts
(model + preprocessor) against the authoritative sha256 hashes recorded in
governance configuration, and only then load them with joblib.

Governance rule enforced here: if the recomputed hash of either artifact
does not exactly match the authoritative value, this module must NOT
deserialize (joblib.load) that artifact at all — a hash mismatch could mean
the file was altered, corrupted, or swapped, and unpickling an unverified
file is itself a risk independent of the governance concern. Instead it
returns a ModelBundle marked unsafe, with a clear reason, for the caller
(app.py) to render as a blocking error screen.

This module never fits, refits, retrains, or mutates the model or
preprocessor. It performs no clinical logic and no prediction.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import sklearn

from backend.governance import ArtifactRecord, Governance


@dataclass(frozen=True)
class ArtifactIntegrityResult:
    """Result of verifying a single artifact file against its authoritative hash."""

    description: str
    path: Path
    file_exists: bool
    expected_sha256: str | None
    actual_sha256: str | None
    hash_match: bool
    error: str | None = None


@dataclass(frozen=True)
class ModelBundle:
    """
    Result of the full load-and-verify process.

    is_safe_to_use is the single flag downstream code must check before
    doing anything with `model` or `preprocessor`. If False, `model` and
    `preprocessor` are guaranteed to be None.
    """

    is_safe_to_use: bool
    model: Any | None
    preprocessor: Any | None
    locked_threshold: float
    model_integrity: ArtifactIntegrityResult
    preprocessor_integrity: ArtifactIntegrityResult
    sklearn_version_active: str
    sklearn_version_required: str
    sklearn_version_matches: bool
    failure_reasons: tuple[str, ...]


def _sha256_of_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(record: ArtifactRecord) -> ArtifactIntegrityResult:
    """Recompute and compare the sha256 of a single artifact. Never loads the file."""
    file_exists = record.path.exists()
    actual = _sha256_of_file(record.path) if file_exists else None

    if not file_exists:
        return ArtifactIntegrityResult(
            description=record.description,
            path=record.path,
            file_exists=False,
            expected_sha256=record.sha256,
            actual_sha256=None,
            hash_match=False,
            error=f"Artifact file not found at {record.path}",
        )

    hash_match = (record.sha256 is not None) and (actual == record.sha256)
    return ArtifactIntegrityResult(
        description=record.description,
        path=record.path,
        file_exists=True,
        expected_sha256=record.sha256,
        actual_sha256=actual,
        hash_match=hash_match,
        error=None if hash_match else "sha256 mismatch against authoritative value",
    )


def load_model_bundle(governance: Governance) -> ModelBundle:
    """
    Verify integrity of the locked model + preprocessor, then load them.

    This is the only function the rest of the application should call to
    obtain the locked model/preprocessor. It always returns a ModelBundle;
    it does not raise for governance-relevant failures (missing file, hash
    mismatch, sklearn version mismatch, load error) — those are reported via
    is_safe_to_use=False and failure_reasons so the UI can render a clear,
    blocking status instead of crashing.
    """
    failure_reasons: list[str] = []

    model_integrity = verify_artifact(governance.model_artifact)
    preprocessor_integrity = verify_artifact(governance.preprocessor_artifact)

    sklearn_version_active = sklearn.__version__
    sklearn_version_matches = sklearn_version_active == governance.required_sklearn_version

    if not model_integrity.hash_match:
        failure_reasons.append(
            f"Model artifact failed integrity verification: {model_integrity.error}"
        )
    if not preprocessor_integrity.hash_match:
        failure_reasons.append(
            f"Preprocessor artifact failed integrity verification: {preprocessor_integrity.error}"
        )
    if not sklearn_version_matches:
        failure_reasons.append(
            "Active scikit-learn version "
            f"({sklearn_version_active}) does not match the required "
            f"version ({governance.required_sklearn_version}) under which "
            "the locked artifacts were serialized. Loading is blocked to "
            "avoid silently invalid deserialization."
        )

    # Do NOT attempt to deserialize either artifact unless both hashes match
    # AND the sklearn version matches. This is a hard governance gate.
    if failure_reasons:
        return ModelBundle(
            is_safe_to_use=False,
            model=None,
            preprocessor=None,
            locked_threshold=governance.locked_threshold,
            model_integrity=model_integrity,
            preprocessor_integrity=preprocessor_integrity,
            sklearn_version_active=sklearn_version_active,
            sklearn_version_required=governance.required_sklearn_version,
            sklearn_version_matches=sklearn_version_matches,
            failure_reasons=tuple(failure_reasons),
        )

    try:
        model = joblib.load(governance.model_artifact.path)
        preprocessor = joblib.load(governance.preprocessor_artifact.path)
    except Exception as exc:  # noqa: BLE001 - we deliberately convert any load error to a safe failure
        failure_reasons.append(f"Failed to load verified artifacts: {exc}")
        return ModelBundle(
            is_safe_to_use=False,
            model=None,
            preprocessor=None,
            locked_threshold=governance.locked_threshold,
            model_integrity=model_integrity,
            preprocessor_integrity=preprocessor_integrity,
            sklearn_version_active=sklearn_version_active,
            sklearn_version_required=governance.required_sklearn_version,
            sklearn_version_matches=sklearn_version_matches,
            failure_reasons=tuple(failure_reasons),
        )

    return ModelBundle(
        is_safe_to_use=True,
        model=model,
        preprocessor=preprocessor,
        locked_threshold=governance.locked_threshold,
        model_integrity=model_integrity,
        preprocessor_integrity=preprocessor_integrity,
        sklearn_version_active=sklearn_version_active,
        sklearn_version_required=governance.required_sklearn_version,
        sklearn_version_matches=sklearn_version_matches,
        failure_reasons=(),
    )

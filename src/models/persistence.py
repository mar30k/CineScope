"""
Model artifact persistence for CineScope.

The fitted model is a plain dictionary of already-processed data:

    {
        "artifact_format_version": int,
        "model_version": str,
        "config_name": str,
        "scoring_strategy": str,
        "weights": {block: float},
        "feature_blocks": [str],
        "total_weight": float,
        "movies": pandas.DataFrame,          # processed metadata
        "features": {block: scipy.sparse},   # raw (un-normalised) matrices
        "encoders": {block: fitted transformer},
        "n_movies": int,
        "created_at": ISO-8601 str,
    }

It is written with ``joblib`` (already a scikit-learn dependency) so no new
packages are required. Saving the processed frame and the feature matrices
means a service can start by loading one file instead of re-running the JSON
parsing, the multi-label encoding and both TF-IDF fits.

This module only moves dictionaries to and from disk; building and rebuilding
the recommender lives in ``src.models.recommender`` so the two do not import
each other.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from src.config import (
    ARTIFACT_DIRECTORY,
    ARTIFACT_FORMAT_VERSION,
    MODEL_ARTIFACT_NAME,
)

REQUIRED_KEYS = (
    "artifact_format_version",
    "model_version",
    "weights",
    "movies",
    "features",
    "n_movies",
)


def default_artifact_path() -> Path:
    """``artifacts/cinescope_model.joblib`` relative to the working directory."""
    return Path(ARTIFACT_DIRECTORY) / MODEL_ARTIFACT_NAME


def save_artifact(payload: dict, path: str | Path | None = None) -> Path:
    """Write a model payload to disk, creating the directory if needed."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict produced by MovieRecommender.to_artifact()")

    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"payload is missing required keys: {missing}")

    target = Path(path) if path is not None else default_artifact_path()
    if target.suffix == "":
        target = target / MODEL_ARTIFACT_NAME

    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, target)
    return target


def load_artifact(path: str | Path | None = None) -> dict:
    """Read a model payload, with explicit errors for the common failures."""
    source = Path(path) if path is not None else default_artifact_path()

    if source.is_dir():
        source = source / MODEL_ARTIFACT_NAME

    if not source.exists():
        raise FileNotFoundError(
            f"No model artifact at '{source}'. Build one with:\n"
            f"    python -m src.models.recommender build"
        )

    try:
        payload = joblib.load(source)
    except Exception as error:                      # corrupt / wrong format
        raise ValueError(
            f"Could not read model artifact '{source}': {error}"
        ) from error

    if not isinstance(payload, dict):
        raise ValueError(
            f"Artifact '{source}' is not a CineScope model (expected a dict, "
            f"got {type(payload).__name__})."
        )

    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"Artifact '{source}' is missing keys: {missing}")

    found = payload.get("artifact_format_version")
    if found != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"Artifact format {found} does not match the supported format "
            f"{ARTIFACT_FORMAT_VERSION}. Rebuild the artifact."
        )

    return payload


def artifact_exists(path: str | Path | None = None) -> bool:
    """True when a loadable artifact file is present."""
    source = Path(path) if path is not None else default_artifact_path()
    if source.is_dir():
        source = source / MODEL_ARTIFACT_NAME
    return source.is_file()


def describe_artifact(path: str | Path | None = None) -> dict:
    """Small summary of an artifact, for health checks and logging."""
    payload = load_artifact(path)
    return {
        "model_version": payload.get("model_version"),
        "config_name": payload.get("config_name"),
        "scoring_strategy": payload.get("scoring_strategy"),
        "weights": payload.get("weights"),
        "n_movies": payload.get("n_movies"),
        "feature_blocks": payload.get("feature_blocks"),
        "created_at": payload.get("created_at"),
    }

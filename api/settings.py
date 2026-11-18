"""
Deployment settings for the CineScope API.

Every value that differs between a laptop and a host is read from the
environment here, so the HTTP layer never parses environment variables itself.

Paths are anchored to the **repository root**, not to the process working
directory. Hosts such as Render do not always start the process in the
directory the command was written for; a relative ``artifacts/...`` would then
resolve somewhere else and the service would fail to find its model.

Environment variables
---------------------
``CORS_ALLOW_ORIGINS``
    Comma-separated browser origins allowed to call the API, e.g.
    ``https://cinescope.vercel.app,http://localhost:5173``. Defaults to the
    two local Vite origins so development works with no configuration.
``CORS_ALLOW_ORIGIN_REGEX``
    Optional regular expression, for hosts that serve preview deployments on
    generated subdomains (``https://.*\\.vercel\\.app``). Empty by default.
``CINESCOPE_ARTIFACT_PATH``
    Where the fitted model lives. Relative values are resolved against the
    repository root. Defaults to ``artifacts/cinescope_model.joblib``.
``CINESCOPE_MODEL_REPO``
    Public Hugging Face Hub repository id (``owner/name``) that holds the
    artifact. Used only when the local file is absent.
``CINESCOPE_MODEL_URL``
    Direct URL to the artifact, for any host that is not the Hub. Takes
    precedence over ``CINESCOPE_MODEL_REPO`` when both are set.
``CINESCOPE_MODEL_FILENAME`` / ``CINESCOPE_MODEL_REVISION``
    File name inside the Hub repository and the revision (branch, tag or
    commit) to download. Default to ``cinescope_model.joblib`` and ``main``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# api/settings.py -> api/ -> repository root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Development defaults: the two origins `npm run dev` can produce.
DEFAULT_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)

DEFAULT_ARTIFACT_RELATIVE = Path("artifacts") / "cinescope_model.joblib"
DEFAULT_CREDITS_RELATIVE = Path("data") / "tmdb-5000" / "tmdb_5000_credits.csv"

DEFAULT_MODEL_FILENAME = "cinescope_model.joblib"
DEFAULT_MODEL_REVISION = "main"

HUGGING_FACE_RESOLVE_URL = "https://huggingface.co/{repo}/resolve/{revision}/{filename}"


def _clean(value: str | None) -> str:
    """Trim whitespace and the quotes some dashboards leave in pasted values."""
    if value is None:
        return ""
    return value.strip().strip('"').strip("'").strip()


def parse_origins(value: str | None) -> list[str]:
    """Parse ``CORS_ALLOW_ORIGINS`` into a normalised origin list.

    Origins are compared literally by the CORS middleware, so a trailing slash
    or duplicated entry would silently fail to match. Empty or whitespace-only
    input keeps the development defaults rather than disabling CORS entirely.
    """
    cleaned = _clean(value)
    if not cleaned:
        return list(DEFAULT_ORIGINS)

    origins: list[str] = []
    for raw in cleaned.split(","):
        origin = _clean(raw).rstrip("/")
        if origin and origin not in origins:
            origins.append(origin)
    return origins or list(DEFAULT_ORIGINS)


def allowed_origins() -> list[str]:
    """Origins allowed to call the API in this process."""
    return parse_origins(os.environ.get("CORS_ALLOW_ORIGINS"))


def allowed_origin_regex() -> str | None:
    """Optional regex for generated preview domains, or ``None``."""
    value = _clean(os.environ.get("CORS_ALLOW_ORIGIN_REGEX"))
    return value or None


def resolve_runtime_path(value: str | None, default_relative: Path) -> Path:
    """Absolute path for a runtime file, anchored to the repository root.

    ``value`` comes from an environment variable when the host needs to place
    the file elsewhere (a mounted volume, for instance). A relative value is
    interpreted from the repository root, which is where the layout documented
    in the README lives; an absolute value is used as given.
    """
    cleaned = _clean(value)
    if not cleaned:
        return (PROJECT_ROOT / default_relative).resolve()

    candidate = Path(cleaned).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def artifact_path() -> Path:
    """Where the model artifact should be, whether or not it exists yet."""
    return resolve_runtime_path(
        os.environ.get("CINESCOPE_ARTIFACT_PATH"), DEFAULT_ARTIFACT_RELATIVE
    )


def credits_path() -> Path:
    """The credits CSV used by the movie-details endpoint."""
    return resolve_runtime_path(
        os.environ.get("CINESCOPE_CREDITS_PATH"), DEFAULT_CREDITS_RELATIVE
    )


@dataclass(frozen=True)
class ModelSource:
    """Where to fetch the artifact from when it is not already on disk."""

    url: str | None
    repo: str | None
    filename: str
    revision: str

    @property
    def configured(self) -> bool:
        return bool(self.url or self.repo)

    def download_url(self) -> str | None:
        """Direct artifact URL, or ``None`` when no source is configured."""
        if self.url:
            return self.url
        if self.repo:
            return HUGGING_FACE_RESOLVE_URL.format(
                repo=self.repo.strip("/"),
                revision=self.revision,
                filename=self.filename,
            )
        return None

    def describe(self) -> str:
        """Human-readable source, safe to log (no query string)."""
        if self.url:
            return "CINESCOPE_MODEL_URL"
        if self.repo:
            return f"Hugging Face Hub repository {self.repo!r} ({self.revision})"
        return "no configured source"


def model_source() -> ModelSource:
    """Artifact source settings for this process."""
    return ModelSource(
        url=_clean(os.environ.get("CINESCOPE_MODEL_URL")) or None,
        repo=_clean(os.environ.get("CINESCOPE_MODEL_REPO")) or None,
        filename=_clean(os.environ.get("CINESCOPE_MODEL_FILENAME"))
        or DEFAULT_MODEL_FILENAME,
        revision=_clean(os.environ.get("CINESCOPE_MODEL_REVISION"))
        or DEFAULT_MODEL_REVISION,
    )

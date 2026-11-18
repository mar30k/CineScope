"""
Obtain the fitted model artifact at startup.

``artifacts/cinescope_model.joblib`` is a build output: ``*.joblib`` is in
``.gitignore``, so a deployment that starts from a clean checkout has no model
on disk. Rebuilding or retraining is explicitly out of scope, so the service
downloads the *existing* file from a public Hugging Face Hub repository (or
any direct URL) and then reuses it.

Behaviour
---------
* If a usable artifact is already at the target path, nothing is downloaded.
  This is checked on every startup, so a warm disk costs one ``stat`` call.
* Otherwise the configured source is fetched into a sibling ``.part`` file and
  atomically moved into place, so an interrupted download can never leave a
  half-written artifact that later starts a corrupt service.
* If there is no local file and no configured source, startup fails with a
  message naming both remedies. A service without a model must not boot.

Only the standard library is used; ``urllib`` is sufficient for one HTTPS GET
and keeps the runtime dependency set unchanged.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from api.settings import ModelSource

logger = logging.getLogger("cinescope.api.artifact")

# The real artifact is ~7 MiB. Anything below this is a truncated download or
# an HTML error page saved under a .joblib name, not a model.
MIN_ARTIFACT_BYTES = 1024

USER_AGENT = "CineScope/1.0"
DEFAULT_TIMEOUT_SECONDS = 60.0
CHUNK_BYTES = 1024 * 1024


class ArtifactUnavailable(RuntimeError):
    """The artifact is neither present locally nor reachable from its source."""


def _safe_url(url: str) -> str:
    """URL without query string, so a token in it never reaches the logs."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _is_usable(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= MIN_ARTIFACT_BYTES
    except OSError:
        return False


def _discard(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:  # pragma: no cover - best effort cleanup
        logger.warning("Could not remove partial download at %s", path)


def ensure_artifact(
    target: Path,
    source: ModelSource,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Make sure ``target`` holds the artifact.

    Returns ``"reused"`` when the file was already present, or
    ``"downloaded"`` when it was fetched during this call.

    Raises
    ------
    ArtifactUnavailable
        The file is absent, no source is configured, the source is unreachable,
        or what came back is too small to be a model.
    """
    if _is_usable(target):
        logger.info("Model artifact found at %s", target)
        return "reused"

    if target.exists():
        # Present but empty/truncated: replacing it is the only way forward.
        logger.warning(
            "Model artifact at %s is empty or truncated (%d bytes); replacing it.",
            target,
            target.stat().st_size,
        )

    url = source.download_url()
    if url is None:
        raise ArtifactUnavailable(
            f"No model artifact at '{target}' and no download source is configured. "
            "Set CINESCOPE_MODEL_REPO to the public Hugging Face Hub repository "
            "(for example 'your-name/cinescope-model'), or set CINESCOPE_MODEL_URL "
            "to a direct artifact URL. Building it locally instead is also "
            "supported: python -m src.models.recommender build"
        )

    safe_url = _safe_url(url)
    logger.info(
        "Model artifact missing at %s; downloading from %s", target, safe_url
    )

    partial = target.with_name(target.name + ".part")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ArtifactUnavailable(
            f"Cannot create the directory for the model artifact at "
            f"'{target.parent}' ({type(error).__name__}: {error})."
        ) from error

    try:
        request = Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
        )
        with urlopen(request, timeout=timeout) as response, partial.open("wb") as handle:
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    except HTTPError as error:
        _discard(partial)
        hint = (
            " The repository or file does not exist, or it is not public."
            if error.code in (401, 403, 404)
            else ""
        )
        raise ArtifactUnavailable(
            f"Could not download the model artifact from {safe_url}: "
            f"HTTP {error.code} ({error.reason}).{hint}"
        ) from error
    except (URLError, OSError, ValueError) as error:
        _discard(partial)
        raise ArtifactUnavailable(
            f"Could not download the model artifact from {safe_url} "
            f"({type(error).__name__}: {error})."
        ) from error

    size = partial.stat().st_size
    if size < MIN_ARTIFACT_BYTES:
        _discard(partial)
        raise ArtifactUnavailable(
            f"The download from {safe_url} produced only {size} bytes, which "
            "cannot be a CineScope model. Check the repository id, file name "
            "and revision."
        )

    try:
        os.replace(partial, target)
    except OSError as error:
        _discard(partial)
        raise ArtifactUnavailable(
            f"Could not move the downloaded artifact into place at '{target}' "
            f"({type(error).__name__}: {error})."
        ) from error

    logger.info(
        "Model artifact downloaded to %s (%.1f MiB)", target, size / 1024 / 1024
    )
    return "downloaded"

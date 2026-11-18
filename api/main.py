"""
CineScope API.

Development:

    uvicorn api.main:app --reload --port 8000 --env-file .env

Hosted (Render and similar):

    python -m api          # reads PORT, binds 0.0.0.0, one worker

The application is deliberately separate from ``src/``: ``src`` holds the
library (data, features, model) and this package holds only the HTTP layer.
Run from the repository root so ``src`` resolves as a namespace package.

Deployment settings are read from the environment in ``api.settings``: allowed
CORS origins, where the model artifact lives and where to download it from.
Paths are anchored to the repository root rather than the working directory.

Layout
------
    api/main.py          app instance, lifespan, error handling
    api/dependencies.py  get_model() accessor
    api/routes.py        HTTP endpoints
    api/schemas.py       response models

Startup
-------
The recommender artifact is loaded **exactly once**, by the lifespan handler
below, and kept on ``app.state.model`` for the life of the process. Handlers
must not call ``MovieRecommender.load()`` themselves: loading costs ~0.2 s and
~7 MB of resident memory, and repeating it per request would be both slow and
wasteful. Route handlers receive the shared instance through
``Depends(get_model)``.

If the artifact is missing, corrupt or built for a different artifact format,
startup raises and the server refuses to serve. A process that cannot
recommend should fail loudly at boot rather than return errors per request.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.artifact_sync import ArtifactUnavailable, ensure_artifact
from api.dependencies import get_model
from api.movie_cast import MovieCastService
from api.routes import router as recommendations_router
from api.search import MovieSearchIndex
from api.settings import (
    allowed_origin_regex,
    allowed_origins,
    artifact_path,
    model_source,
)
from api.tmdb_images import TmdbImageService
from src.config import CONFIG_NAME, MODEL_NAME, MODEL_VERSION, SCORING_STRATEGY
from src.models.recommender import MovieRecommender

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
logger = logging.getLogger("cinescope.api")


class ModelLoadError(RuntimeError):
    """The recommender could not be loaded, so the API cannot start."""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once on startup, release it on shutdown."""
    logger.info("Loading %s %s ...", MODEL_NAME, MODEL_VERSION)

    # The artifact is a build output and is not in version control, so on a
    # hosted instance it is normally fetched once from the public model
    # repository. An artifact already on disk is reused and never re-fetched.
    artifact = artifact_path()
    source = model_source()
    try:
        status = ensure_artifact(artifact, source)
    except ArtifactUnavailable as error:
        raise ModelLoadError(
            f"CineScope cannot start: {error}"
        ) from error
    logger.info(
        "Model artifact %s (%s)", "reused" if status == "reused" else "downloaded",
        artifact,
    )

    try:
        model = MovieRecommender.load(artifact)
    except Exception as error:
        # Any failure here - corrupt joblib, artifact format mismatch, wrong
        # scikit-learn version - must abort startup. Re-raised as one typed
        # error so the cause is obvious in the boot log instead of surfacing
        # later as a 500 on the first request.
        raise ModelLoadError(
            f"CineScope cannot start: the recommendation model at '{artifact}' "
            f"failed to load ({type(error).__name__}: {error}). "
            f"Rebuild it with: python -m src.models.recommender build"
        ) from error

    app.state.model = model
    app.state.cast_service = MovieCastService.load()
    app.state.image_service = TmdbImageService.from_environment()
    # Normalized titles, tokens and years are computed once here so a search
    # request never rebuilds them.
    app.state.search_index = MovieSearchIndex.build(model.movies)
    logger.info("Search index ready: %d titles.", app.state.search_index.n_titles)

    logger.info(
        "Model ready: %d movies, config %r, strategy %s, weights %s",
        model.n_movies, CONFIG_NAME, SCORING_STRATEGY, model.weights,
    )
    logger.info("CORS origins: %s", ", ".join(allowed_origins()))
    if allowed_origin_regex():
        logger.info("CORS origin regex: %s", allowed_origin_regex())
    if app.state.cast_service.n_movies:
        logger.info(
            "Cast metadata ready: %d movies with credits.",
            app.state.cast_service.n_movies,
        )
    else:
        logger.warning(
            "Cast metadata unavailable. Details will return an empty cast list."
        )
    if app.state.image_service.enabled:
        logger.info("TMDB image enrichment enabled.")
    else:
        logger.warning(
            "TMDB image enrichment disabled: TMDB_API_KEY is not set. "
            "The API will serve without artwork."
        )

    try:
        yield
    finally:
        app.state.image_service = None
        app.state.cast_service = None
        app.state.search_index = None
        app.state.model = None
        logger.info("Model released; shutdown complete.")


app = FastAPI(
    title=f"{MODEL_NAME} API",
    version=MODEL_VERSION,
    description=(
        "Content-based movie recommendations over a 4,803-film catalogue. "
        "Ranking uses Strategy B (fixed-denominator weighted content "
        "agreement) with the L feature configuration."
    ),
    lifespan=lifespan,
)

# Localhost stays the default so `npm run dev` works with no configuration;
# a deployed frontend is added through CORS_ALLOW_ORIGINS. Preflight requests
# are answered by the middleware itself, before any route is matched.
ALLOWED_ORIGINS = allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=allowed_origin_regex(),
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["Accept", "Content-Type"],
    max_age=600,
)

app.include_router(recommendations_router)


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, error: Exception) -> JSONResponse:
    """Last-resort handler: log the details, return a generic JSON error.

    The traceback stays in the server log; the client sees a stable shape and
    no internal paths or stack frames.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


@app.get("/", tags=["meta"])
def root() -> dict:
    """Liveness endpoint: identifies the service and confirms it is serving."""
    return {
        "service": MODEL_NAME,
        "version": MODEL_VERSION,
        "status": "ok",
        "message": f"{MODEL_NAME} API is running.",
    }

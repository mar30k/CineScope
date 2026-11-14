"""
CineScope API.

    uvicorn api.main:app --reload --port 8000

The application is deliberately separate from ``src/``: ``src`` holds the
library (data, features, model) and this package holds only the HTTP layer.
Run from the project root so ``src`` resolves as a namespace package.

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
from fastapi.responses import JSONResponse

from api.dependencies import get_model
from api.routes import router as recommendations_router
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
    try:
        model = MovieRecommender.load()
    except Exception as error:
        # Any failure here - missing file, corrupt joblib, artifact format
        # mismatch - must abort startup. Re-raised as one typed error so the
        # cause is obvious in the boot log instead of surfacing later as a
        # 500 on the first request.
        raise ModelLoadError(
            f"CineScope cannot start: the recommendation model failed to "
            f"load ({type(error).__name__}: {error}). "
            f"Rebuild it with: python -m src.models.recommender build"
        ) from error

    app.state.model = model
    logger.info(
        "Model ready: %d movies, config %r, strategy %s, weights %s",
        model.n_movies, CONFIG_NAME, SCORING_STRATEGY, model.weights,
    )

    try:
        yield
    finally:
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

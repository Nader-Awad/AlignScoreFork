from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from .alignscore_runner import AlignScoreRunner
from .config import ServiceSettings
from .schemas import ScoreRequest, ScoreResponse


logger = logging.getLogger("alignscore.service")


def create_app() -> FastAPI:
    settings = ServiceSettings.from_env()

    app = FastAPI(
        title="AlignScore Service",
        description="REST API wrapper around AlignScore factual consistency metric.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        # Ensure model loads at startup for first request latency.
        runner = AlignScoreRunner.instance(settings)
        logger.info(
            "AlignScore loaded: device=%s, eval_mode=%s, batch=%d",
            runner.device,
            runner.evaluation_mode,
            runner.default_batch_size,
        )

    @app.get("/healthz", tags=["health"])
    async def healthz() -> Dict[str, Any]:
        runner = AlignScoreRunner.instance(settings)
        return {
            "status": "ok",
            "device": runner.device,
            "evaluation_mode": runner.evaluation_mode,
            "batch_size": runner.default_batch_size,
        }

    @app.post("/score", response_model=ScoreResponse, tags=["alignscore"])
    async def score(request: ScoreRequest) -> ScoreResponse:
        runner = AlignScoreRunner.instance(settings)
        try:
            scores = await runner.score_async(
                (item.context for item in request.items),
                (item.claim for item in request.items),
                batch_size=request.batch_size,
                evaluation_mode=request.evaluation_mode,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return ScoreResponse(
            scores=scores,
            evaluation_mode=request.evaluation_mode or runner.evaluation_mode,
            batch_size=request.batch_size or runner.default_batch_size,
            device=runner.device,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request, exc):  # type: ignore[override]
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("service.app:app", host="127.0.0.1", port=9000, reload=False)

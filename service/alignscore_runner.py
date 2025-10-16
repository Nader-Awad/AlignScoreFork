from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterable, List

import torch
from alignscore import AlignScore
from starlette.concurrency import run_in_threadpool

from .config import ServiceSettings


class AlignScoreRunner:
    """Singleton-style AlignScore wrapper with thread-safe batching."""

    _instance: "AlignScoreRunner | None" = None
    _lock = threading.Lock()

    def __init__(self, settings: ServiceSettings) -> None:
        if not settings.model_path.exists():
            raise FileNotFoundError(
                f"AlignScore checkpoint not found at {settings.model_path}"
            )

        device = settings.device
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if device == "mps":
            # 1.12.1 lacks several kernels on MPS; enable CPU fallback.
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            if not torch.backends.mps.is_available():
                device = "cpu"

        self._device = device
        self._evaluation_mode = settings.evaluation_mode
        self._default_batch_size = settings.batch_size

        backbone = "roberta-large"
        if "base" in settings.model_path.stem.lower():
            backbone = "roberta-base"

        self._scorer = AlignScore(
            model=backbone,
            batch_size=settings.batch_size,
            device=self._device,
            ckpt_path=str(settings.model_path),
            evaluation_mode=self._evaluation_mode,
        )

    @classmethod
    def instance(cls, settings: ServiceSettings | None = None) -> "AlignScoreRunner":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    if settings is None:
                        settings = ServiceSettings.from_env()
                    cls._instance = cls(settings)
        return cls._instance

    @property
    def device(self) -> str:
        return self._device

    @property
    def evaluation_mode(self) -> str:
        return self._evaluation_mode

    @property
    def default_batch_size(self) -> int:
        return self._default_batch_size

    def score(
        self,
        contexts: Iterable[str],
        claims: Iterable[str],
        *,
        batch_size: int | None = None,
        evaluation_mode: str | None = None,
    ) -> List[float]:
        ctx_list = list(contexts)
        claim_list = list(claims)
        if len(ctx_list) != len(claim_list):
            raise ValueError("contexts and claims must have the same length")

        if not ctx_list:
            return []

        # AlignScore's batch_size/evaluation_mode are set at init; override if provided.
        if batch_size and batch_size != self._scorer.batch_size:
            self._scorer.batch_size = batch_size
        if evaluation_mode and evaluation_mode != self._scorer.evaluation_mode:
            self._scorer.evaluation_mode = evaluation_mode

        scores = self._scorer.score(contexts=ctx_list, claims=claim_list)
        return [float(s) for s in scores]

    async def score_async(
        self,
        contexts: Iterable[str],
        claims: Iterable[str],
        *,
        batch_size: int | None = None,
        evaluation_mode: str | None = None,
    ) -> List[float]:
        return await run_in_threadpool(
            self.score,
            contexts,
            claims,
            batch_size=batch_size,
            evaluation_mode=evaluation_mode,
        )

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_PATH = Path("models/AlignScore-large.ckpt")
DEFAULT_DEVICE = "auto"  # auto -> prefer mps if available
DEFAULT_EVAL_MODE = "nli_sp"
DEFAULT_BATCH_SIZE = 8


@dataclass(frozen=True)
class ServiceSettings:
    model_path: Path
    device: str
    evaluation_mode: str
    batch_size: int

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        model_path = Path(
            os.getenv("ALIGNSCORE_MODEL_PATH", str(DEFAULT_MODEL_PATH))
        ).expanduser()
        device = os.getenv("ALIGNSCORE_DEVICE", DEFAULT_DEVICE).lower()
        evaluation_mode = os.getenv("ALIGNSCORE_EVAL_MODE", DEFAULT_EVAL_MODE)
        batch_size_str = os.getenv("ALIGNSCORE_BATCH_SIZE")
        try:
            batch_size = int(batch_size_str) if batch_size_str else DEFAULT_BATCH_SIZE
            if batch_size <= 0:
                raise ValueError
        except ValueError:
            batch_size = DEFAULT_BATCH_SIZE

        return cls(
            model_path=model_path,
            device=device,
            evaluation_mode=evaluation_mode,
            batch_size=batch_size,
        )


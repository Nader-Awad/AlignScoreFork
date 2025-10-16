from __future__ import annotations

import os
from pathlib import Path

import torch
from alignscore import AlignScore


def main() -> None:
    """
    Minimal AlignScore smoke test.

    Loads the checkpoint at models/AlignScore-large.ckpt (override with
    ALIGNSCORE_MODEL_PATH) and scores a single reference/claim pair.
    """
    ckpt_path = Path(
        os.getenv("ALIGNSCORE_MODEL_PATH", "models/AlignScore-large.ckpt")
    ).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"AlignScore checkpoint not found at {ckpt_path}. "
            "Download it or set ALIGNSCORE_MODEL_PATH."
        )

    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    scorer = AlignScore(
        model="roberta-large",
        batch_size=4,
        device=device,
        ckpt_path=str(ckpt_path),
        evaluation_mode="nli_sp",
    )

    contexts = [
        "The NBA season of 1975-76 was the 30th season of the National Basketball Association."
    ]
    claims = [
        "The 1975-76 season of the NBA was its 30th season."
    ]

    scores = scorer.score(contexts=contexts, claims=claims)
    print(f"AlignScore device: {device}")
    print(f"Sample score: {scores[0]:.4f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, validator


class ScoreItem(BaseModel):
    context: str = Field(..., description="Reference text / source document.")
    claim: str = Field(..., description="Candidate text whose consistency is evaluated.")

    @validator("context", "claim")
    def _strip(cls, value: str) -> str:
        v = value.strip()
        if not v:
            raise ValueError("Text must not be empty.")
        return v


class ScoreRequest(BaseModel):
    items: List[ScoreItem] = Field(..., min_items=1)
    evaluation_mode: Optional[str] = Field(
        None, description="Override AlignScore evaluation mode (default nli_sp)."
    )
    batch_size: Optional[int] = Field(
        None,
        description="Override AlignScore batch size.",
        gt=0,
    )


class ScoreResponse(BaseModel):
    scores: List[float]
    evaluation_mode: str
    batch_size: int
    device: str


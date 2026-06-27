# api/schemas.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class RoutingDecision(str, Enum):
    AUTO_CLASSIFY = "auto_classify"
    CLASSIFY_LOG  = "classify_log"
    HUMAN_REVIEW  = "human_review"


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Input text to classify (Hinglish/English)",
        examples=["yaar tu bahut bura insaan hai"]
    )
    threshold: Optional[float] = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Classification threshold for HOF label"
    )

    @field_validator('text')
    @classmethod
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class ScoreDetail(BaseModel):
    NOT: float = Field(description="Probability of non-hate content")
    HOF: float = Field(description="Probability of hate/offensive content")


class PredictResponse(BaseModel):
    text          : str
    label         : str
    confidence    : float
    scores        : ScoreDetail
    routing       : RoutingDecision
    text_length   : int
    model_version : str = "muril-base-cased-v1"


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of texts to classify (max 32)"
    )
    threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)


class BatchPredictResponse(BaseModel):
    results      : list[PredictResponse]
    total        : int
    hof_count    : int
    not_count    : int
    review_count : int


class HealthResponse(BaseModel):
    status       : str
    model_loaded : bool
    model_name   : str
    device       : str
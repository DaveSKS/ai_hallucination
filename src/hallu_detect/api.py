"""Optional FastAPI endpoint for response verification."""

import os
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .nli import NliClassifier, load_label_map
from .pipeline import HallucinationVerifier


class VerifyRequest(BaseModel):
    response: str
    documents: list[str | dict[str, str]] = Field(default_factory=list)


@lru_cache(maxsize=1)
def _verifier() -> HallucinationVerifier:
    model = os.getenv("HALLU_MODEL")
    if not model:
        raise RuntimeError("set HALLU_MODEL to a fine-tuned NLI checkpoint before starting the API")
    return HallucinationVerifier(NliClassifier(model, label_map=load_label_map()))


app = FastAPI(title="Hallu Detect", version="0.1.0")


@app.post("/verify")
def verify(request: VerifyRequest) -> dict:
    return _verifier().verify(request.response, request.documents)

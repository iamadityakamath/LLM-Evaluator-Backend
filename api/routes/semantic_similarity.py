from typing import Any

from fastapi import APIRouter, HTTPException, Request
from sentence_transformers import util

router = APIRouter()

@router.post("/semantic_similarity")
def semantic_similarity(payload: dict[str, Any], request: Request):
    text1 = payload.get("text1")
    text2 = payload.get("text2")

    if text1 is None or text2 is None:
        raise HTTPException(
            status_code=400,
            detail="Payload must include text1 and text2.",
        )

    model = request.app.state.sentence_model
    embeddings1 = model.encode(text1, convert_to_tensor=True)
    embeddings2 = model.encode(text2, convert_to_tensor=True)
    similarity_score = float(util.pytorch_cos_sim(embeddings1, embeddings2))

    return {"similarity_score": similarity_score}
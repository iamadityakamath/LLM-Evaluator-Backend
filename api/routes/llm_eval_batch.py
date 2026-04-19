from typing import Any
from fastapi import APIRouter, HTTPException
from llmevalkit import Evaluator

router = APIRouter()
evaluator = Evaluator(provider="none", preset="math")

@router.post("/llm_eval_batch")
def llm_eval_batch(payload: dict[str, Any]):
    questions = payload.get("questions", [])
    grounded_answers = payload.get("grounded_answers", [])
    model_answers = payload.get("model_answers", [])

    if not (questions and grounded_answers and model_answers):
        raise HTTPException(
            status_code=400,
            detail="Payload must include questions, grounded_answers, and model_answers (all lists).",
        )
    if not (len(questions) == len(grounded_answers) == len(model_answers)):
        raise HTTPException(
            status_code=400,
            detail="questions, grounded_answers, and model_answers must be the same length.",
        )

    results = []
    for q, g, m in zip(questions, grounded_answers, model_answers):
        result = evaluator.evaluate(question=q, answer=m, context=g)
        results.append({
            "question": q,
            "model_answer": m,
            "grounded_answer": g,
            "summary": result.summary(),
        })
    return {"results": results}

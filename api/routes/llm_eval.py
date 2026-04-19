from typing import Any

from fastapi import APIRouter, HTTPException
from llmevalkit import Evaluator

router = APIRouter()

evaluator = Evaluator(provider="none", preset="math")


@router.post("/llm_eval")
def llm_eval(payload: dict[str, Any]):
    questions = payload.get("questions", [])
    grounded_answers = payload.get("grounded_answers", [])
    model_answer = payload.get("model_answer")

    if not questions or not grounded_answers or model_answer is None:
        raise HTTPException(
            status_code=400,
            detail="Payload must include questions, grounded_answers, and model_answer.",
        )

    result = evaluator.evaluate(
        question=questions[0],
        answer=model_answer,
        context=grounded_answers[0],
    )

    return {"bleu": result.metrics['bleu'].name, 
            "bleu_score": result.metrics['bleu'].score,
            "rouge": result.metrics['rouge'].name,
            "rouge_score": result.metrics['rouge'].score,
            "token_overlap": result.metrics['token_overlap'].name,
            "token_overlap_score": result.metrics['token_overlap'].score,
            "keyword_coverage": result.metrics['keyword_coverage'].name,
            "keyword_coverage_score": result.metrics['keyword_coverage'].score,
            "answer_length": result.metrics['answer_length'].name,
            "answer_length_score": result.metrics['answer_length'].score,   
            "readability": result.metrics['readability'].name,
            "readability_score": result.metrics['readability'].score
            }
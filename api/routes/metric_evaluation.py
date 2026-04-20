from typing import Any
from fastapi import APIRouter, Request
from api.routes.semantic_similarity import semantic_similarity
from api.routes.llm_eval_batch import llm_eval_batch
from fastapi.responses import StreamingResponse
import csv
import io

router = APIRouter()

@router.post("/metric_evaluation")
def metric_evaluation(payload: dict[str, Any], request: Request):
    records = payload.get("records", [])
    if not records:
        return {"error": "No records provided"}

    questions = [r["question"] for r in records]
    model_answers = [r["model_answer"] for r in records]
    grounded_answers = [r["grounded_answer"] for r in records]

    # LLM evaluation batch
    eval_payload = {
        "questions": questions,
        "grounded_answers": grounded_answers,
        "model_answers": model_answers
    }
    eval_results = llm_eval_batch(eval_payload)
    eval_metrics = eval_results["results"]

    # Semantic similarity for each record
    similarities = []
    for m, g in zip(model_answers, grounded_answers):
        sim_payload = {"text1": m, "text2": g}
        sim_result = semantic_similarity(sim_payload, request)
        similarities.append(sim_result["similarity_score"])

    # Merge results
    merged = []
    for i, r in enumerate(records):
        merged.append({
            **r,
            **eval_metrics[i],
            "similarity_score": similarities[i]
        })

    return {"results": merged}

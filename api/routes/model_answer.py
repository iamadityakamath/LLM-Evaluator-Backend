import os
from typing import Any
from fastapi import APIRouter, HTTPException
import openai
from dotenv import load_dotenv
from groq import Groq
from concurrent.futures import ThreadPoolExecutor, as_completed

router = APIRouter()
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
groq_client = Groq()

def get_openai_answer(prompt: str, question: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message["content"].strip()

def get_groq_answer(prompt: str, question: str) -> str:
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.7,
        max_completion_tokens=3000,
        top_p=1,
        stream=False,
        stop=None
    )
    return completion.choices[0].message.content.strip()

@router.post("/model_answer")
def model_answer(payload: dict[str, Any]):
    question = payload.get("question")
    prompt = payload.get("prompt")
    model_type = payload.get("model_type", "openai")
    if not question or not prompt:
        raise HTTPException(
            status_code=400,
            detail="Payload must include question and prompt."
        )
    try:
        if model_type == "groq":
            answer = get_groq_answer(prompt, question)
        else:
            answer = get_openai_answer(prompt, question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model_answer_batch")
def model_answer_batch(payload: dict[str, Any]):
    questions = payload.get("questions", [])
    prompt = payload.get("prompt")
    model_type = payload.get("model_type", "openai")
    max_workers = 3
    if not questions or not prompt:
        raise HTTPException(
            status_code=400,
            detail="Payload must include questions (list) and prompt."
        )
    
    def call_model(q):
        try:
            if model_type == "groq":
                return get_groq_answer(prompt, q)
            else:
                return get_openai_answer(prompt, q)
        except Exception as e:
            return e

    results = [None] * len(questions)
    failed_indices = []

    # First pass: parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(call_model, q): i for i, q in enumerate(questions)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                res = future.result()
                if isinstance(res, Exception):
                    failed_indices.append(idx)
                else:
                    results[idx] = res
            except Exception:
                failed_indices.append(idx)

    # Retry failed ones (serially, but could also do in parallel)
    for idx in failed_indices[:]:
        try:
            res = call_model(questions[idx])
            if isinstance(res, Exception):
                continue
            results[idx] = res
            failed_indices.remove(idx)
        except Exception:
            continue

    return {
        "answers": results,
        "failed_indices": failed_indices
    }

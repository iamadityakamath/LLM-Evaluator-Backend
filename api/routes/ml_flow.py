from typing import Any

import mlflow
import openai
import pandas as pd

from fastapi import APIRouter

router = APIRouter()


@router.post("/ML_flow")
def ml_flow(payload: dict[str, Any]):
    eval_df = pd.DataFrame({
      "inputs": payload.get("questions", []),
      "ground_truth": payload.get("ground_truth", [])
      })
    return {"received": payload}

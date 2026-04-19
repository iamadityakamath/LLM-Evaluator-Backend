from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

from api.routes.health import router as health_router
from api.routes.llm_eval import router as llm_eval_router
from api.routes.llm_eval_batch import router as llm_eval_batch_router
from api.routes.semantic_similarity import router as semantic_similarity_router
from api.routes.ml_flow import router as ml_flow_router
from api.routes.model_answer import router as model_answer_router

app = FastAPI(title="FastAPI Vercel Template")

@app.on_event("startup")
def load_models():
    app.state.sentence_model = SentenceTransformer("./my_local_model")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(llm_eval_router)
app.include_router(llm_eval_batch_router)
app.include_router(semantic_similarity_router)
app.include_router(ml_flow_router)
app.include_router(model_answer_router)


@app.get("/")
def root():
    return {"message": "Hello from FastAPI"}

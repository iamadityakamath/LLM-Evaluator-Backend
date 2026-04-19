from fastapi import FastAPI

from routes.health import router as health_router

app = FastAPI(title="FastAPI Vercel Template")
app.include_router(health_router)


@app.get("/")
def root():
    return {"message": "Hello from FastAPI"}

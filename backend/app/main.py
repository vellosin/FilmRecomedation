from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.movies import router as movies_router
from app.routes.recommendations import router as recommendations_router
from app.routes.training import router as training_router


app = FastAPI(
    title="Movie Recommender API",
    version="0.1.0",
    description="API local para treino e recomendacao de filmes com base no dataset TMDB.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(training_router)
app.include_router(movies_router)
app.include_router(recommendations_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

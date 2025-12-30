from fastapi import FastAPI
from pydantic import BaseModel
from recommender import MovieRecommender
import asyncio
from functools import partial
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Emotion Based Movie Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = MovieRecommender()

class RecommendationRequest(BaseModel):
    text: str
    limit: int = 5

class RecommendationResponse(BaseModel):
    emotion: str
    movies: list[str]
    cached: bool

@app.post('/recommender',
          response_model=RecommendationResponse
          )
async def recommed(req : RecommendationRequest):
    loop = asyncio.get_event_loop()

    movies, emotion, cached = await loop.run_in_executor(
        None,
        partial(recommender.get_recommendations, req.text, req.limit)
    )

    return {
        "emotion" : emotion,
        "movies" : movies,
        "cached" : cached
    }



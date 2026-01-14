from fastapi import APIRouter, Request
from recommender import MovieRecommender
import asyncio
from functools import partial
from schemas import RecommendationRequest, RecommendationResponse

router = APIRouter()

@router.post('/recommender', response_model=RecommendationResponse)
async def recommed(req : RecommendationRequest, request: Request):
    recommender = request.app.state.recommender
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
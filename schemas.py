from pydantic import BaseModel

class RecommendationRequest(BaseModel):
    text: str
    limit: int = 5

class RecommendationResponse(BaseModel): 
    emotion: str
    movies: list[str]
    cached: bool
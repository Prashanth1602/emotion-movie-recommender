from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from recommender import MovieRecommender
from config import settings
from redisclient import RedisCache
from routes import router

import logging

# Configure logging to show INFO logs (essential for Docker/Northflank)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Redis
    cache = RedisCache(host=settings.redis_host, port=settings.redis_port)
    try:
        cache.client.ping()
        logger.info("Successfully connected to Redis.")
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {e}. Caching will be disabled.")

    # Initialize Recommender
    app.state.recommender = MovieRecommender(csv_path=settings.MOVIES_CSV_PATH, cache_client=cache)
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

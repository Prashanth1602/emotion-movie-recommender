import redis
import os

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_password = os.getenv("REDIS_PASSWORD", None)

redis_client = redis.Redis(
    host=redis_host,
    port=int(redis_port),
    password=redis_password,
    ssl=True,
    decode_responses=True
)

class RedisCache:
    def get_emotion(self, text: str):
        key = f"emotion:{text}"
        return redis_client.get(key)

    def set_emotion(self, text: str, emotion: str, ttl=3600):
        key = f"emotion:{text}"
        redis_client.setex(key, ttl, emotion)


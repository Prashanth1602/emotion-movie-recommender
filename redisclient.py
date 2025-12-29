import redis

redis_client = redis.Redis(
    host = "localhost",
    port = 6379,
    db = 0,
    decode_responses=True
)

class RedisCache:
    def get_emotion(self, text: str):
        key = f"emotion:{text}"
        return redis_client.get(key)

    def set_emotion(self, text: str, emotion: str, ttl=3600):
        key = f"emotion:{text}"
        redis_client.setex(key, ttl, emotion)


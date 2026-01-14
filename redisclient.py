import redis

class RedisCache:
    def __init__(self, host, port):
        self.client = redis.Redis(
            host=host,
            port=port,
            decode_responses=True,
            socket_timeout=1,
            socket_connect_timeout=1
        )

    def get_emotion(self, text: str):
        key = f"emotion:{text}"
        return self.client.get(key)

    def set_emotion(self, text: str, emotion: str, ttl=3600):
        key = f"emotion:{text}"
        self.client.setex(key, ttl, emotion)


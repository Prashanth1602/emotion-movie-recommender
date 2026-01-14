from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MOVIES_CSV_PATH: str = "movies.csv"
    class Config:
        env_file = ".env"
    
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379

settings = Settings()

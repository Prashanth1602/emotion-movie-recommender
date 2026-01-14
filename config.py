from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MOVIES_CSV_PATH: str = "movies.csv"
    class Config:
        env_file = ".env"
    
    redis_host: str 
    redis_port: int 

settings = Settings()

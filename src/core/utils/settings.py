from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mongo_host: str
    mongo_port: int = 27017
    mongo_db: str

    mongo_user: str
    mongo_password: str
    mongo_auth_source: str = "admin"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

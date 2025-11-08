# env vars (pydantic Settings)

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DB_URL: str
    S3_BUCKET: str
    AWS_REGION: str = "us-west-2"
    BEDROCK_EMBED_MODEL: str = "amazon.titan-embed-text-v2:0"
    BEDROCK_CHAT_MODEL: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"

    class Config:
        env_file = ".env"

settings = Settings()

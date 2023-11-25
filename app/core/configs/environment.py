"""
Module to load all Environment variables
"""

from pydantic_settings import BaseSettings


class Environment(BaseSettings):
    """
    Environment, add the variable and its type here matching the .env file
    """

    # APPLICATION
    APPLICATION_HOST: str = "localhost"
    APPLICATION_PORT: int = 8000
    APPLICATION_NAME: str = "GreyWolf Service"
    MODEL_MINIMAL_AGE: int = 5

    # DATABASE
    DATABASE_URL: str = "localhost:5432"
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5432
    DATABASE_USER: str = "user"
    DATABASE_PASSWORD: str = "password"
    DATABASE_NAME: str = "test"
    ENVIRONMENT: str = "test"

    # S3
    BUCKET_BASE_URL: str = "localhost"
    BUCKET_ACCESS_KEY_ID: str = "test"
    BUCKET_SECRET_KEY: str = "test"
    BUCKET_NAME: str = "test"
    BUCKET_ACL: str = "test"
    BUCKET_URL_EXPIRES_IN_SECONDS: int = 0

    # GREY WOLF
    GWO_EPOCH: int = 500
    GWO_POP_SIZE: int = 10
    TEST_SIZE: float = 0.25

    # PROPERTY API
    PROPERTY_API_URL: str

    # RABBIT
    RBMQ_HOST: str
    RBMQ_USER: str
    RBMQ_PASS: str
    RBMQ_PORT: int
    RBMQ_EXCHANGE: str
    RBMQ_VHOST: str
    PREFETCH_VALUE: int

    # QUEUES
    TRAIN_MODEL_CHANNEL: str

    class Config:
        """Load config file"""

        env_file = ".env"
        extra='ignore'

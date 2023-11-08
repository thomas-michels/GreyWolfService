from pydantic import BaseModel, Field
from datetime import datetime


class ModelHistory(BaseModel):
    model_id: int = Field(example=123)
    epoch: int = Field(example=123)
    mse: float = Field(default=1, example=123)
    params: dict = Field(default={})


class ModelHistoryInDB(ModelHistory):
    id: int = Field(example=123)
    created_at: datetime = Field(example=str(datetime.now()))
    updated_at: datetime = Field(example=str(datetime.now()))

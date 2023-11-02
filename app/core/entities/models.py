from pydantic import BaseModel, Field
from datetime import datetime


class Model(BaseModel):
    path: str = Field(default=None, example="path")
    neighborhood_encoder: str = Field(default=None, example="path")
    one_hot_encoder: str = Field(default=None, example="path")
    x_min_max: str = Field(default=None, example="path")
    y_min_max: str = Field(default=None, example="path")
    mse: float = Field(default=0, example=123)


class ModelInDB(Model):
    id: int = Field(example=123)
    created_at: datetime = Field(example=str(datetime.now()))
    updated_at: datetime = Field(example=str(datetime.now()))

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from .model_histories import ModelHistoryInDB

class ModelStatus(str, Enum):
    SCHEDULED: str = "SCHEDULED"
    TRAINING: str = "TRAINING"
    READY: str = "READY"
    ERROR: str = "ERROR"


class Model(BaseModel):
    name: Optional[str] = Field(default="", example="test")
    status: Optional[ModelStatus] = Field(default=ModelStatus.SCHEDULED, example=ModelStatus.SCHEDULED)
    path: str = Field(default="", example="path")
    neighborhood_encoder: str = Field(default="", example="path")
    one_hot_encoder: str = Field(default="", example="path")
    x_min_max: str = Field(default="", example="path")
    y_min_max: str = Field(default="", example="path")
    mse: float = Field(default=0, example=123)
    gwo_params: Optional[dict] = Field(default={})


class ModelInDB(Model):
    id: int = Field(example=123)
    created_at: datetime = Field(example=str(datetime.now()))
    updated_at: datetime = Field(example=str(datetime.now()))


class ModelWithHistory(BaseModel):
    id: int = Field(example=123)
    name: Optional[str] = Field(default="", example="test")
    mse: float = Field(default=0, example=123)
    gwo_params: Optional[dict] = Field(default={})
    created_at: datetime = Field(example=str(datetime.now()))
    updated_at: datetime = Field(example=str(datetime.now()))
    history: List[ModelHistoryInDB] = Field(default=[])

from typing import Optional
from pydantic import BaseModel, Field


class Property(BaseModel):
    rooms: int = Field(example=2)
    bathrooms: int = Field(example=2)
    parking_space: int = Field(example=2)
    size: int = Field(example=100)
    neighborhood_name: str = Field(example="viktor konder")
    flood_quota: Optional[float] = Field(default=None, example=123)


class PredictedProperty(BaseModel):
    property: Property
    predicted_price: float = Field(example=123)
    mse: float = Field(example=123)

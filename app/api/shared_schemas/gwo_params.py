from pydantic import BaseModel, Field


class GWOParams(BaseModel):
    epochs: int = Field(default=10, example=10, gt=0)
    population_size: int = Field(default=10, example=10, gt=9)
    min_max_iter: int = Field(default=10, example=10, gt=0)
    max_max_iter: int = Field(default=50, example=10, gt=0)
    min_neurons: int = Field(default=10, example=10, gt=0)
    max_neurons: int = Field(default=100, example=10, gt=0)
    hidden_layers: int = Field(default=2, example=10, gt=0)
    min_learning_rate: float = Field(default=0.0001, example=10, gt=0)
    max_learning_rate: float = Field(default=0.1, example=10, gt=0)
    min_momentum: float = Field(default=0.001, example=10, gt=0)
    max_momentum: float = Field(default=1, example=10, gt=0)
    min_batch_size: int = Field(default=16, example=10, gt=0)
    max_batch_size: int = Field(default=256, example=10, gt=0)

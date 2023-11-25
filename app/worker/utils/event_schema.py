"""
    Module for event schemas
"""
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic.types import UUID4


class EventSchema(BaseModel):
    """
    Event Schema
    """

    id: UUID4 = Field(example="16f8ddc6-3697-4b90-a5c5-1b60e26de6dc")
    origin: str = Field(example="Queue")
    sent_to: str = Field(example="Queue")
    payload: dict | str = Field(default={})
    created_at: datetime = Field(example="2022-06-04 22:13:19.332981")
    updated_at: datetime = Field(example="2022-06-04 22:13:19.332981")

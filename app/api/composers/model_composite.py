from fastapi import Depends
from app.core.services import ModelServices
from app.core.db import PGConnection
from app.core.db.repositories import (
    ModelRepository,
    PropertyRepository,
    ModelHistoryRepository,
)


def model_composer(
    conn: PGConnection = Depends(PGConnection),
) -> ModelServices:
    model_repository = ModelRepository(connection=conn)
    property_repository = PropertyRepository()
    model_history_repository = ModelHistoryRepository(connection=conn)
    service = ModelServices(
        model_repository=model_repository,
        property_repository=property_repository,
        model_history_repository=model_history_repository,
    )
    return service

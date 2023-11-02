from fastapi import Depends
from app.core.services import ModelServices
from app.core.db import PGConnection
from app.core.db.repositories import ModelRepository, PropertyRepository


async def model_composer(
    conn: PGConnection = Depends(PGConnection),
) -> ModelServices:
    model_repository = ModelRepository(connection=conn)
    property_repository = PropertyRepository()
    service = ModelServices(
        model_repository=model_repository, property_repository=property_repository
    )
    return service

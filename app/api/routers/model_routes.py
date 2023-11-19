from fastapi import APIRouter, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import List
from app.api.composers import model_composer
from app.api.shared_schemas import GWOParams
from app.core.services import ModelServices
from app.core.entities import (
    Property,
    PredictedProperty,
    ModelWithHistory,
    SummarizedModel,
)
from app.core.configs import get_logger


router = APIRouter(prefix="/models", tags=["Models"])
_logger = get_logger(__name__)


@router.post("/train")
async def train_model(
    gwo_params: GWOParams,
    worker: BackgroundTasks,
    name: str = "Grey Wolf V2",
    services: ModelServices = Depends(model_composer),
):
    model_in_db = await services.pre_create_model(name=name, gwo_params=gwo_params)
    if model_in_db:
        worker.add_task(services.train_and_save_model, model_in_db)
        return JSONResponse(
            status_code=201,
            content=jsonable_encoder(
                {
                    "message": "A new model will be trained!",
                    "data": model_in_db.model_dump(),
                }
            ),
        )

    else:
        return JSONResponse(
            status_code=400,
            content=jsonable_encoder(
                {"message": "You can't train a new model now, await some minutes!"}
            ),
        )


@router.post("/predict/price", responses={200: {"model": PredictedProperty}})
async def predict_price(
    property: Property, model_id: int=None, services: ModelServices = Depends(model_composer)
):
    try:
        predicted_property = await services.predict_price(model_id=model_id, property=property)

        if predicted_property:
            return JSONResponse(
                status_code=200,
                content=jsonable_encoder(predicted_property.model_dump()),
            )

        else:
            return JSONResponse(
                status_code=400,
                content=jsonable_encoder({"message": "Some error happen"}),
            )

    except Exception as error:
        _logger.error(f"Error on predict_price: {str(error)}")
        return JSONResponse(
            status_code=400,
            content=jsonable_encoder({"message": f"Some error happen: {str(error)}"}),
        )


@router.get("", responses={200: {"model": List[ModelWithHistory]}})
async def get_trained_models(
    page: int = Query(default=1, gt=0),
    page_size: int = Query(default=10, gt=0),
    services: ModelServices = Depends(model_composer),
):
    try:
        models = await services.search_models(page=page, page_size=page_size)

        if models:
            return JSONResponse(
                status_code=200,
                content=jsonable_encoder([model.model_dump() for model in models]),
            )

        else:
            return JSONResponse(
                status_code=400,
                content=jsonable_encoder({"message": "Some error happen"}),
            )

    except Exception as error:
        _logger.error(f"Error on get_trained_models: {str(error)}")
        return JSONResponse(
            status_code=400,
            content=jsonable_encoder({"message": f"Some error happen: {str(error)}"}),
        )


@router.get("/{model_id}", responses={200: {"model": SummarizedModel}})
async def get_model_by_id(
    model_id: int, services: ModelServices = Depends(model_composer)
):
    try:
        model = await services.search_model_by_id(id=model_id)

        if model:
            return JSONResponse(
                status_code=200, content=jsonable_encoder(model.model_dump())
            )

        else:
            return JSONResponse(
                status_code=404, content=jsonable_encoder({"message": "Not found"})
            )

    except Exception as error:
        _logger.error(f"Error on get_model_by_id: {str(error)}")
        return JSONResponse(
            status_code=400,
            content=jsonable_encoder({"message": f"Some error happen: {str(error)}"}),
        )

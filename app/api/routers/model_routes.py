from fastapi import APIRouter, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import List
from app.api.composers import model_composer
from app.core.services import ModelServices
from app.core.entities import Property, PredictedProperty, ModelWithHistory

router = APIRouter(prefix="/models", tags=["Models"])


@router.post("/train")
async def train_model(
    worker: BackgroundTasks,
    services: ModelServices = Depends(model_composer)
):
    
    if await services.check_minimal_age():
        worker.add_task(services.train_and_save_model)

        return JSONResponse(status_code=202, content=jsonable_encoder({"message": "A new model will be trained!"}))
    
    else:

        return JSONResponse(status_code=400, content=jsonable_encoder({"message": "You can't train a new model now, await some minutes!"}))

@router.post("/predict/price", responses={200: {"model": PredictedProperty}})
async def predict_price(
    property: Property,
    services: ModelServices = Depends(model_composer)
):
    try:
        predicted_property = await services.predict_price(property=property)

        if predicted_property:
            return JSONResponse(status_code=200, content=jsonable_encoder(predicted_property.model_dump()))
        
        else:
            return JSONResponse(status_code=400, content=jsonable_encoder({"message": "Some error happen"}))

    except Exception as error:
        return JSONResponse(status_code=400, content=jsonable_encoder({"message": f"Some error happen: {str(error)}"}))

@router.get("", responses={200: {"model": List[ModelWithHistory]}})
async def get_trained_models(
    page: int = Query(default=1, gt=0),
    page_size: int = Query(default=10, gt=0),
    services: ModelServices = Depends(model_composer)
):
    try:
        models = await services.search_models(page=page, page_size=page_size)

        if models:
            return JSONResponse(status_code=200, content=jsonable_encoder([model.model_dump() for model in models]))
        
        else:
            return JSONResponse(status_code=400, content=jsonable_encoder({"message": "Some error happen"}))

    except Exception as error:
        return JSONResponse(status_code=400, content=jsonable_encoder({"message": f"Some error happen: {str(error)}"}))

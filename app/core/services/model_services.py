from typing import List
from app.core.db.repositories import (
    ModelRepository,
    PropertyRepository,
    ModelHistoryRepository,
)
from app.core.services.train_services import TrainServices
from app.core.services.prediction_services import PredictionServices
from app.core.services.preprocessing_services import PreProcessingServices
from app.core.entities import (
    Model,
    ModelInDB,
    Property,
    PredictedProperty,
    ModelStatus,
    ModelWithHistory,
)
from app.core.configs import get_environment, get_logger
from datetime import datetime, timezone

_env = get_environment()
_logger = get_logger(__name__)


class ModelServices:
    def __init__(
        self,
        model_repository: ModelRepository,
        property_repository: PropertyRepository,
        model_history_repository: ModelHistoryRepository,
    ) -> None:
        self.__model_repository = model_repository
        self.__property_repository = property_repository
        self.__model_history_repository = model_history_repository

    async def train_and_save_model(self) -> ModelInDB:
        model = Model(name="GREY WOLF V1")

        model_in_db = await self.__model_repository.create(model=model)

        file_url = self.__property_repository.get_all_properties()
        if not file_url or not model_in_db:
            return

        await self.__model_repository.update_status(
            new_status=ModelStatus.TRAINING, model_id=model_in_db.id
        )

        preprocessing = PreProcessingServices(file_url=file_url)

        preprocessing.normalize()

        preprocessing.filter_best_characteristics()

        preprocessing.apply_label_encoder()

        preprocessing.apply_one_hot_encoder()

        preprocessing.scale()

        preprocessing.split()

        train_services = TrainServices(
            model_in_db=model_in_db,
            x_properties_train=preprocessing.x_properties_train,
            y_properties_train=preprocessing.y_properties_train,
            x_properties_test=preprocessing.x_properties_test,
            y_properties_test=preprocessing.y_properties_test,
        )

        mse, model_path = train_services.train()

        model_in_db.path = model_path
        model_in_db.mse = mse

        preprocessing.save(model_in_db)

        is_updated = await self.__model_repository.update(model_in_db=model_in_db)

        if is_updated:
            await self.__model_repository.update_status(
                new_status=ModelStatus.READY, model_id=model_in_db.id
            )

        else:
            await self.__model_repository.update_status(
                new_status=ModelStatus.ERROR, model_id=model_in_db.id
            )

        return model_in_db

    async def predict_price(self, property: Property) -> PredictedProperty:
        prediction_services = PredictionServices()

        latest_model = await self.search_latest()

        preprocessing = PreProcessingServices(model=latest_model)

        property_array = [
            property.rooms,
            property.bathrooms,
            property.size,
            property.parking_space,
            property.neighborhood_name,
            property.flood_quota,
        ]

        normalized_property = preprocessing.normalize_property(
            property_array=property_array
        )

        price = prediction_services.predict(
            bucket_path=latest_model.path, normalized_property=normalized_property
        )

        predicted_property = PredictedProperty(
            property=property, predicted_price=price, mse=latest_model.mse
        )

        (
            predicted_property.predicted_price,
            predicted_property.mse,
        ) = preprocessing.desnormalize(
            predicted_property.predicted_price, predicted_property.mse
        )

        return predicted_property

    async def search_latest(self) -> ModelInDB:
        model_in_db = await self.__model_repository.select_latest()

        return model_in_db

    async def search_models(self, page: int, page_size: int) -> List[ModelWithHistory]:
        models = await self.__model_repository.select_models(
            page=page, page_size=page_size
        )

        models_id = []

        for model in models:
            if model.status == ModelStatus.READY:
                models_id.append(model.id)

        histories = self.__model_history_repository.select_model_histories_by_model_id(models_id=models_id)

        models_with_history = []

        for model in models:
            models_with_history.append(ModelWithHistory(
                id=model.id,
                name=model.name,
                mse=model.mse,
                gwo_params=model.gwo_params,
                created_at=model.created_at,
                updated_at=model.updated_at,
                history=histories.get(model.id, [])
            ))

        return models_with_history

    async def check_minimal_age(self) -> bool:
        latest_model = await self.search_latest()
        if latest_model:
            current_datetime = datetime.now(timezone.utc)

            model_age = ((latest_model.created_at - current_datetime).seconds) / 60

            if model_age <= _env.MODEL_MINIMAL_AGE:
                return False

        return True

from app.core.db.repositories import ModelRepository, PropertyRepository
from app.core.services.train_services import TrainServices
from app.core.services.prediction_services import PredictionServices
from app.core.services.preprocessing_services import PreProcessingServices
from app.core.entities import Model, ModelInDB, Property, PredictedProperty
from app.core.configs import get_environment, get_logger
from datetime import datetime, timezone

_env = get_environment()
_logger = get_logger(__name__)


class ModelServices:
    def __init__(
        self, model_repository: ModelRepository, property_repository: PropertyRepository
    ) -> None:
        self.__model_repository = model_repository
        self.__property_repository = property_repository

    async def train_and_save_model(self) -> ModelInDB:
        file_url = self.__property_repository.get_all_properties()
        if not file_url:
            return

        preprocessing = PreProcessingServices(file_url=file_url)

        preprocessing.normalize()

        preprocessing.filter_best_characteristics()

        preprocessing.apply_label_encoder()

        preprocessing.apply_one_hot_encoder()

        preprocessing.scale()

        preprocessing.split()

        train_services = TrainServices(
            x_properties_train=preprocessing.x_properties_train,
            y_properties_train=preprocessing.y_properties_train,
            x_properties_test=preprocessing.x_properties_test,
            y_properties_test=preprocessing.y_properties_test
        )

        mse, model_path = train_services.train()

        model = Model(
            path=model_path,
            mse=mse
        )

        preprocessing.save(model)

        model_in_db = await self.__model_repository.create(model=model)

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
            property.flood_quota
        ]

        normalized_property = preprocessing.normalize_property(property_array=property_array)

        price = prediction_services.predict(
            bucket_path=latest_model.path,
            normalized_property=normalized_property
        )

        predicted_property = PredictedProperty(
            property=property,
            predicted_price=price,
            mse=latest_model.mse
        )

        predicted_property.predicted_price, predicted_property.mse = preprocessing.desnormalize(
            predicted_property.predicted_price,
            predicted_property.mse
        )

        return predicted_property

    async def search_latest(self) -> ModelInDB:
        model_in_db = await self.__model_repository.select_latest()

        return model_in_db

    async def check_minimal_age(self) -> bool:
        latest_model = await self.search_latest()
        if latest_model:
            current_datetime = datetime.now(timezone.utc)

            model_age = ((current_datetime - latest_model.created_at).seconds) / 60

            if model_age <= _env.MODEL_MINIMAL_AGE:
                return False

        return True
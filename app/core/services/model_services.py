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
    SummarizedModel
)
from app.api.shared_schemas import GWOParams
from app.core.configs import get_environment, get_logger

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

    def pre_create_model(self, name: str, gwo_params: GWOParams) -> ModelInDB:
        lb_neurons = [gwo_params.min_neurons] * gwo_params.hidden_layers
        ub_neurons = [gwo_params.max_neurons] * gwo_params.hidden_layers

        name = name

        if gwo_params.property_type:
            name = f"{name} - {gwo_params.property_type}"

        model = Model(
            name=name,
            epochs=gwo_params.epochs,
            population_size=gwo_params.population_size,
            gwo_params={
                "lb": [gwo_params.min_max_iter, gwo_params.min_learning_rate, gwo_params.min_momentum, gwo_params.min_batch_size] + lb_neurons,
                "ub": [gwo_params.max_max_iter, gwo_params.max_learning_rate, gwo_params.max_momentum, gwo_params.max_batch_size] + ub_neurons,
            }
        )

        model.gwo_params = {
            "max_iter": [model.gwo_params["lb"][0], model.gwo_params["ub"][0]],
            "learning_rate": [model.gwo_params["lb"][1], model.gwo_params["ub"][1]],
            "momentum": [model.gwo_params["lb"][2], model.gwo_params["ub"][2]],
            "batch_size": [model.gwo_params["lb"][3], model.gwo_params["ub"][3]],
            "hidden_layer_sizes": [model.gwo_params["lb"][4:], model.gwo_params["ub"][4:]],
            "lb": model.gwo_params["lb"],
            "ub": model.gwo_params["ub"],
            "minmax": "min",
            "property_type": gwo_params.property_type
        }

        model_in_db = self.__model_repository.create(model=model)

        return model_in_db

    def train_and_save_model(self, model_in_db: ModelInDB) -> ModelInDB:

        model_in_db = self.__model_repository.select_by_id(id=model_in_db.id)

        if model_in_db.status != ModelStatus.SCHEDULED:
            return model_in_db

        file_url = self.__property_repository.get_all_properties()
        if not file_url or not model_in_db:
            return

        self.__model_repository.update_status(
            new_status=ModelStatus.TRAINING, model_id=model_in_db.id
        )

        try:
            preprocessing = PreProcessingServices(file_url=file_url)

            preprocessing.normalize()

            preprocessing.filter_best_characteristics(only=model_in_db.gwo_params.get("property_type"))

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

            is_updated = self.__model_repository.update(model_in_db=model_in_db)

            if is_updated:
                self.__model_repository.update_status(
                    new_status=ModelStatus.READY, model_id=model_in_db.id
                )

            else:
                self.__model_repository.update_status(
                    new_status=ModelStatus.ERROR, model_id=model_in_db.id
                )

        except Exception as error:
            _logger.error(f"Error on train_model: {str(error)}")
            self.__model_repository.update_status(
                new_status=ModelStatus.ERROR, model_id=model_in_db.id
            )

        return model_in_db

    def predict_price(self, model_id: int, property: Property) -> PredictedProperty:
        prediction_services = PredictionServices()

        latest_model = self.search_complete_model_by_id(id=model_id)

        if not latest_model:
            return

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

    def search_latest(self) -> ModelInDB:
        model_in_db = self.__model_repository.select_latest()

        return model_in_db

    def search_models(self, page: int, page_size: int) -> List[ModelWithHistory]:
        models = self.__model_repository.select_models(
            page=page, page_size=page_size
        )

        models_id = []

        for model in models:
            if model.status == ModelStatus.READY and model.mse > 0:
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

    def search_model_by_id(self, id: int) -> SummarizedModel:
        model = self.__model_repository.select_by_id(id=id)

        if model.status == ModelStatus.TRAINING:
            remaining_time_in_seconds = self.__model_repository.select_remaining_time(model_id=id)
            model.remaining_time_in_seconds = remaining_time_in_seconds

        return model

    def delete_model_by_id(self, id: int) -> bool:
        self.__model_history_repository.delete_by_model_id(model_id=id)
        return self.__model_repository.delete_by_id(id=id)

    def search_complete_model_by_id(self, id: int) -> ModelInDB:
        if id:
            model = self.__model_repository.select_complete_by_id(id=id)

        else:
            model = self.__model_repository.select_latest()

        return model

    def search_statistics(self) -> dict:
        return self.__model_repository.select_model_statistics()

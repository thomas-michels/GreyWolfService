from typing import List, Tuple
from mealpy.swarm_based.GWO import BaseGWO
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import tempfile
from app.core.configs import get_environment, get_logger
from app.core.entities import ModelHistory, ModelInDB
from app.core.db import PGConnection
from app.core.db.repositories import ModelHistoryRepository
from app.api.dependencies import Bucket


_env = get_environment()
_logger = get_logger(__name__)


class TrainServices:
    def __init__(
        self,
        model_in_db: ModelInDB,
        x_properties_train: np.array,
        y_properties_train: np.array,
        x_properties_test: np.array,
        y_properties_test: np.array,
    ) -> None:
        self.x_properties_train = x_properties_train
        self.y_properties_train = y_properties_train
        self.x_properties_test = x_properties_test
        self.y_properties_test = y_properties_test
        self.params = {
            "fit_func": self.fitness_func,
            "lb": [10, 10, 10, 0.0001, 0.001, 16],
            "ub": [50, 100, 100, 0.1, 1, 256],
            "minmax": "min",
        }
        self.mse = 1
        self.epoch = 1
        self.model_in_db = model_in_db
        self.__model_history_repository = ModelHistoryRepository(connection=PGConnection())
        self.__save_gwo_params()

    def train(self) -> Tuple[float, str]:
        _logger.info(f"Starting train at {datetime.now()}")

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_model_file:
            self.find_best_fitness_with_gwo()
            self.save(file=temp_model_file.name)

            bucket_path = self.__get_model_path()

            Bucket.save_file(bucket_path, temp_model_file.name)

            _logger.info(f"Model trained at {datetime.now()}")

        return self.mse, bucket_path

    def find_best_fitness_with_gwo(self):
        start = datetime.now()
        _logger.info(f"Starting GWO - {start}")
        gwo = BaseGWO(self.params, _env.GWO_EPOCH, _env.GWO_POP_SIZE)
        best_position, best_fitness = gwo.solve()

        self.best_position = best_position
        self.best_fitness = best_fitness
        _logger.info(f"Finished GWO - {((datetime.now() - start).seconds) / 60} minutes!")

    def fitness_func(self, solution: tuple) -> float:
        max_iter = int(solution[0])
        hidden_layer_sizes = [int(solution[1]), int(solution[2])]
        learning_rate = solution[3]
        momentum = solution[4]
        batch_size = int(solution[5])

        model = Sequential()

        for hidden_units in hidden_layer_sizes:
            model.add(Dense(units=hidden_units, activation="relu"))

        model.add(Dense(units=1, activation="relu"))

        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)

        model.compile(loss="mean_absolute_error", optimizer=optimizer)

        model.fit(self.x_properties_train, self.y_properties_train, epochs=max_iter, verbose=0)

        predictions = model.predict(self.x_properties_test, batch_size=batch_size, verbose=0)
        predictions = np.squeeze(predictions)

        mse = mean_absolute_error(self.y_properties_test, predictions)

        self.__save_history(
            mse=mse,
            max_iter=max_iter,
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate=learning_rate,
            momentum=momentum,
            batch_size=batch_size
        )

        if mse < self.mse:
            self.mse = mse
            self.model = model

        return mse if mse else 1
    
    def save(self, file: str):
        self.model.save(file)

    def __get_model_path(self) -> str:
        now = datetime.now()

        return f"GWO/GWO_{now.year}-{now.month}-{now.day}-{now.hour}:{now.minute}.h5"

    def __save_history(
            self,
            mse: float,
            max_iter: int,
            hidden_layer_sizes: List[int],
            learning_rate: float,
            momentum: float,
            batch_size: int
        ):
        history = ModelHistory(
            model_id=self.model_in_db.id,
            epoch=self.epoch,
            mse=mse,
            params={
                "max_iter": max_iter,
                "hidden_layer_sizes": hidden_layer_sizes,
                "learning_rate": learning_rate,
                "momentum": momentum,
                "batch_size": batch_size,
            }
        )

        self.__model_history_repository.create(model_history=history)
        self.epoch += 1

    def __save_gwo_params(self):
        self.model_in_db.gwo_params = {
            "max_iter": [self.params["lb"][0], self.params["ub"][0]],
            "hidden_layer_sizes": [self.params["lb"][1:-3], self.params["ub"][1:-3]],
            "learning_rate": [self.params["lb"][-3], self.params["ub"][-3]],
            "momentum": [self.params["lb"][-2], self.params["ub"][-2]],
            "batch_size": [self.params["lb"][-1], self.params["ub"][-1]],
            "lb": self.params["lb"],
            "ub": self.params["ub"],
            "minmax": "min",
        }

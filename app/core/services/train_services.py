from typing import Tuple
from mealpy.swarm_based.GWO import BaseGWO
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import tempfile
from app.core.configs import get_environment, get_logger
from app.api.dependencies import Bucket


_env = get_environment()
_logger = get_logger(__name__)


class TrainServices:
    def __init__(
        self,
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
            "lb": [10, 19, 19, 19, 19, 0.0001, 0.001, 16],
            "ub": [50, 117, 117, 117, 117, 0.9, 1, 256],
            "minmax": "min",
        }
        self.mse = 1

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
        hidden_layer_sizes = (int(solution[1]), int(solution[2]), int(solution[3]), int(solution[4]))
        learning_rate = solution[5]
        momentum = solution[6]
        batch_size = int(solution[7])

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

        if mse < self.mse:
            self.mse = mse
            self.model = model

        return mse if mse else 1
    
    def save(self, file: str):
        self.model.save(file)

    def __get_model_path(self) -> str:
        now = datetime.now()

        return f"GWO/GWO_{now.year}-{now.month}-{now.day}-{now.hour}:{now.minute}.h5"

from typing import Tuple
from mealpy.swarm_based.GWO import BaseGWO
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
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
            "lb": [50, 19, 19, 19, 0.0001, 0.001, 16],
            "ub": [500, 117, 117, 117, 0.1, 1, 256],
            "minmax": "min",
        }

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
        gwo = BaseGWO(self.params, _env.GWO_EPOCH, _env.GWO_POP_SIZE)
        best_position, best_fitness = gwo.solve()

        self.best_position = best_position
        self.best_fitness = best_fitness

    def fitness_func(self, solution: tuple, file=None) -> float:
        max_iter = int(solution[0])
        hidden_layer_sizes = (int(solution[1]), int(solution[2]), int(solution[3]))
        learning_rate = solution[4]
        momentum = solution[5]
        batch_size = int(solution[6])

        self.model = Sequential()

        for hidden_units in hidden_layer_sizes:
            self.model.add(Dense(units=hidden_units, activation="relu"))

        self.model.add(Dense(units=1, activation="relu"))

        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)

        self.model.compile(loss="mse", optimizer=optimizer)

        self.model.fit(self.x_properties_train, self.y_properties_train, epochs=max_iter)

        predictions = self.model.predict(self.x_properties_test, batch_size=batch_size)
        predictions = np.squeeze(predictions)

        self.mse = mean_squared_error(self.y_properties_test, predictions)

        return self.mse if self.mse else 0
    
    def save(self, file: str):
        self.model.save(file)

    def __get_model_path(self) -> str:
        now = datetime.now()

        return f"GWO/GWO_{now.year}-{now.month}-{now.day}-{now.hour}:{now.minute}.h5"

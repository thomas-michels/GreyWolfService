from typing import Tuple
import pandas as pd
import tempfile
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from app.api.dependencies import Bucket
from app.core.configs import get_environment
from app.core.entities import ModelInDB
from datetime import datetime
import requests

_env = get_environment()


class PreProcessingServices:
    def __init__(self, model: ModelInDB=None, file_url: str=None) -> None:
        self.label_encoder_neighborhood = LabelEncoder()
        self.onehot_encoder_properties = ColumnTransformer(
            transformers=[("OneHot", OneHotEncoder(), [4])], remainder="passthrough"
        )
        self.x_min_max_scaler = MinMaxScaler()
        self.y_min_max_scaler = MinMaxScaler()

        if file_url:
            self.__file_url = file_url
            self.load_dataframe()

        else:
            self.__load_label_encoder(model.neighborhood_encoder)
            self.__load_one_hot_encoder(model.one_hot_encoder)
            self.__load_x_min_max_scaler(model.x_min_max)
            self.__load_y_min_max_scaler(model.y_min_max)

    def load_dataframe(self):
        self.dataframe = pd.read_csv(
            self.__file_url, delimiter=";", quotechar="|", index_col="id"
        )

    def normalize(self):
        self.dataframe["main_type"] = self.dataframe["type"].apply(
            self.__normalize_type
        )

        self.dataframe["flood_quota"] = self.dataframe["flood_quota"].apply(
            self.__convert_flood_quota
        )

        self.dataframe["security"] = self.dataframe["flood_quota"].apply(
            self.__check_security
        )

        self.dataframe["price"] = self.dataframe["price"].apply(lambda v: v / 1000)

    def filter_best_characteristics(self):
        self.sell_dataframe = self.dataframe[self.dataframe["modality_name"] == "venda"]

        self.x_properties = self.sell_dataframe.loc[
            :,
            [
                "rooms",
                "bathrooms",
                "size",
                "parking_space",
                "neighborhood_name",
                "security",
            ],
        ]
        self.y_properties = self.sell_dataframe.loc[:, ["price"]]

    def apply_label_encoder(self):
        self.x_properties.iloc[:, 4] = self.label_encoder_neighborhood.fit_transform(
            self.x_properties.iloc[:, 4]
        )

    def apply_one_hot_encoder(self):
        self.x_properties = self.onehot_encoder_properties.fit_transform(
            self.x_properties
        ).toarray()

    def scale(self):
        self.x_properties_finished = self.x_min_max_scaler.fit_transform(
            self.x_properties, self.y_properties
        )
        self.y_properties_finished = self.y_min_max_scaler.fit_transform(
            self.y_properties
        )

    def split(self):
        (
            self.x_properties_train,
            self.x_properties_test,
            self.y_properties_train,
            self.y_properties_test,
        ) = train_test_split(
            self.x_properties_finished, self.y_properties_finished, test_size=_env.TEST_SIZE, random_state=0
        )

    def save(self, model: ModelInDB) -> dict:
        
        model.neighborhood_encoder = self.__save_label_encoder()
        model.one_hot_encoder = self.__save_one_hot_encoder()
        model.x_min_max = self.__save_x_min_max()
        model.y_min_max = self.__save_y_min_max()

    def normalize_property(self, property_array: list) -> np.array:
        flood_quota = self.__convert_flood_quota(property_array[-1])
        property_array[-1] = self.__check_security(flood_quota)
        
        property_array[4] = self.label_encoder_neighborhood.transform([property_array[4]])[0]

        property_array = self.onehot_encoder_properties.transform([property_array]).toarray()[0]

        property_array = self.x_min_max_scaler.transform([property_array])[0]
        return property_array
    
    def desnormalize(self, price: float, mse: float) -> Tuple[float, float]:
        prices = self.y_min_max_scaler.inverse_transform([[price, mse]])

        return round(prices[0][0], 2) * 1000, round(prices[0][1], 2) * 1000

    def __save_label_encoder(self) -> str:
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_label_encoder:

            joblib.dump(self.label_encoder_neighborhood, temp_label_encoder.name)

            bucket_path = self.__get_model_path(model_name="NEIGHBORHOOD_ENCODER", type="joblib")

            Bucket.save_file(bucket_path, temp_label_encoder.name)

        return bucket_path

    def __save_one_hot_encoder(self) -> str:
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as one_hot_encoder:

            joblib.dump(self.onehot_encoder_properties, one_hot_encoder.name)

            bucket_path = self.__get_model_path(model_name="ONE_HOT_ENCODER", type="joblib")

            Bucket.save_file(bucket_path, one_hot_encoder.name)

        return bucket_path

    def __save_x_min_max(self) -> str:
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_x_min_max:

            joblib.dump(self.x_min_max_scaler, temp_x_min_max.name)

            bucket_path = self.__get_model_path(model_name="X_MIN_MAX", type="joblib")

            Bucket.save_file(bucket_path, temp_x_min_max.name)

        return bucket_path

    def __save_y_min_max(self) -> str:
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_y_min_max:
            joblib.dump(self.y_min_max_scaler, temp_y_min_max.name)

            bucket_path = self.__get_model_path(model_name="Y_MIN_MAX", type="joblib")

            Bucket.save_file(bucket_path, temp_y_min_max.name)

        return bucket_path
    
    def __load_label_encoder(self, bucket_path: str):
        sign_url = Bucket.get_presigned_url(path=bucket_path)

        response = requests.get(sign_url)
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_model_file:
            with open(temp_model_file.name, 'wb') as file:
                file.write(response.content)

            self.label_encoder_neighborhood = joblib.load(temp_model_file.name)

    def __load_one_hot_encoder(self, bucket_path: str):
        sign_url = Bucket.get_presigned_url(path=bucket_path)

        response = requests.get(sign_url)
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_model_file:
            with open(temp_model_file.name, 'wb') as file:
                file.write(response.content)

            self.onehot_encoder_properties = joblib.load(temp_model_file.name)

    def __load_x_min_max_scaler(self, bucket_path: str):
        sign_url = Bucket.get_presigned_url(path=bucket_path)

        response = requests.get(sign_url)
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_model_file:
            with open(temp_model_file.name, 'wb') as file:
                file.write(response.content)

            self.x_min_max_scaler = joblib.load(temp_model_file.name)

    def __load_y_min_max_scaler(self, bucket_path: str):
        sign_url = Bucket.get_presigned_url(path=bucket_path)

        response = requests.get(sign_url)
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as temp_model_file:
            with open(temp_model_file.name, 'wb') as file:
                file.write(response.content)

            self.y_min_max_scaler = joblib.load(temp_model_file.name)

    def __normalize_type(self, value):
        if value == "penthouse":
            return "apartamento"

        elif value == "flat":
            return "apartamento"

        elif value == "loft":
            return "apartamento"

        elif value == "sobrado":
            return "casa"

        elif value == "geminada":
            return "casa"

        elif value == "condominium":
            return "casa"

        elif value == "kitnet":
            return "casa"

        return value

    def __convert_flood_quota(self, value):
        return 21 if pd.isna(value) else value

    def __check_security(self, value):
        if value < 8.13:
            return 1

        elif 8.14 < value < 9.15:
            return 2

        elif 9.16 < value < 12.6:
            return 3

        else:
            return 4

    def __get_model_path(self, model_name: str, type: str) -> str:
        now = datetime.now()

        return f"Encoders/{model_name.upper()}_{now.year}-{now.month}-{now.day}-{now.second}.{type}"

from app.api.dependencies import Bucket
from keras.models import load_model
import numpy as np
import requests
import tempfile


class PredictionServices:

    def predict(self, bucket_path: str, normalized_property: np.array) -> float:
        
        self.__load_model(bucket_path=bucket_path)

        list_property = list(normalized_property)

        prediction = self.trained_model.predict([list_property], batch_size=64)

        return prediction[0][0]
    
    def __load_model(self, bucket_path: str):

        sign_url = Bucket.get_presigned_url(path=bucket_path)

        response = requests.get(sign_url)
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_model_file:
            with open(temp_model_file.name, 'wb') as file:
                file.write(response.content)

            self.trained_model = load_model(temp_model_file.name)

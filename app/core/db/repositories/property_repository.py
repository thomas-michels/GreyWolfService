import requests
from app.core.configs import get_environment, get_logger
import time

_env = get_environment()
_logger = get_logger(__name__)


class PropertyRepository:

    def __init__(self) -> None:
        pass

    def get_all_properties(self, model_id: int) -> str:
        for i in range(5):
            try:
                url = f"{_env.PROPERTY_API_URL}/properties/export/csv"

                response = requests.get(url=url, params={"model_id": model_id})

                response.raise_for_status()

                json = response.json()

                return json["file_url"]

            except Exception as error:
                _logger.error(f"Error on get_all_properties: {str(error)} - Retry: {i+1}")
                time.sleep(2)

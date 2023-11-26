from app.callbacks.callback_interface.callback_base import Callback
from app.core.db import PGConnection
from app.worker.utils.event_schema import EventSchema
from app.api.composers import model_composer
from app.core.configs import get_logger
from app.core.services import ModelServices
from app.core.entities import ModelInDB

_logger = get_logger(__name__)


class TrainModelCallback(Callback):

    def __init__(self, conn: PGConnection) -> None:
        super().__init__(conn)
        self.__model_services: ModelServices = model_composer(conn=self.conn)

    def handle(self, message: EventSchema) -> bool:
        try:
            model_in_db = ModelInDB(**message.payload)
            _logger.info(f"Model -> {model_in_db.model_dump_json(indent=4)}")
        
            trained_model = self.__model_services.train_and_save_model(model_in_db=model_in_db)

            _logger.info(f"Training - #{trained_model.id}")

            if trained_model:
                _logger.info(f"New model trained - #{trained_model.id}")

            return True

        except Exception as error:
            _logger.error(f"Error on TrainModelCallback - Error: {str(error)}")
            return False

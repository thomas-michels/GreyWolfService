from app.core.db import DBConnection
from app.core.db.repositories.base_repository import Repository
from app.core.entities import ModelHistory, ModelHistoryInDB
from app.core.configs import get_logger
import json

_logger = get_logger(__name__)


class ModelHistoryRepository(Repository):
    def __init__(self, connection: DBConnection) -> None:
        super().__init__(connection)

    def create(self, model_history: ModelHistory) -> ModelHistoryInDB:
        query = """--sql
        INSERT INTO
            public.model_histories
            (model_id, epoch, mse, params, created_at, updated_at)
        VALUES(%(model_id)s, %(epoch)s, %(mse)s, %(params)s, NOW(), NOW())
        RETURNING id, model_id, epoch, mse, params, created_at, updated_at;
        """
        try:
            result = self.conn.fetch_with_retry(sql_statement=query, values={
                "model_id": model_history.model_id,
                "epoch": model_history.epoch,
                "mse": model_history.mse,
                "params": json.dumps(model_history.params)
            })
            self.conn.commit()

            if result:
                return ModelHistoryInDB(
                    id=result["id"],
                    model_id=result["model_id"],
                    epoch=result["epoch"],
                    mse=result["mse"],
                    params=result["params"],
                    created_at=result["created_at"],
                    updated_at=result["updated_at"],
                )

        except Exception as error:
            _logger.error(f"Error: {str(error)}")

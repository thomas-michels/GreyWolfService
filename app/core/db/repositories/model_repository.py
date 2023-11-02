from app.core.db import DBConnection
from app.core.db.repositories.base_repository import Repository
from app.core.entities import Model, ModelInDB
from app.core.configs import get_logger

_logger = get_logger(__name__)


class ModelRepository(Repository):
    def __init__(self, connection: DBConnection) -> None:
        super().__init__(connection)

    async def create(self, model: Model) -> ModelInDB:
        query = """
        INSERT
            INTO
            development.models
        ("path", created_at, updated_at, x_min_max_scaler, y_min_max_scaler, neighborhood_encoder, one_hot_encoder, mse)
        VALUES(%(path)s, NOW(), NOW(), %(x_min_max_scaler)s, %(y_min_max_scaler)s, %(neighborhood_encoder)s, %(one_hot_encoder)s, %(mse)s)
        RETURNING id, "path", x_min_max_scaler, y_min_max_scaler, neighborhood_encoder, one_hot_encoder, mse, created_at, updated_at;
        """
        try:
            result = self.conn.execute(sql_statement=query, values={
                "path": model.path,
                "x_min_max_scaler": model.x_min_max,
                "y_min_max_scaler": model.y_min_max,
                "neighborhood_encoder": model.neighborhood_encoder,
                "one_hot_encoder": model.one_hot_encoder,
                "mse": model.mse,
            })
            self.conn.commit()

            if result:
                return ModelInDB(**result)

        except Exception as error:
            _logger.error(f"Error: {str(error)}")

    async def select_latest(self) -> ModelInDB:
        query = """
        SELECT
            id,
            "path",
            x_min_max_scaler AS x_min_max,
            y_min_max_scaler AS y_min_max,
            neighborhood_encoder,
            one_hot_encoder,
            mse,
            created_at,
            updated_at
        FROM
            public.models m
        ORDER BY
            created_at DESC
        LIMIT 1 OFFSET 0;
        """
        try:
            self.conn.execute(sql_statement=query)
            result = self.conn.fetch()

            if result:
                return ModelInDB(**result)

        except Exception as error:
            _logger.error(f"Error: {str(error)}")

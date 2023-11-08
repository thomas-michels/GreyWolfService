import json
from typing import List
from app.core.db import DBConnection
from app.core.db.repositories.base_repository import Repository
from app.core.entities import Model, ModelInDB, ModelStatus, SummarizedModel
from app.core.configs import get_logger

_logger = get_logger(__name__)


class ModelRepository(Repository):
    def __init__(self, connection: DBConnection) -> None:
        super().__init__(connection)

    async def create(self, model: Model) -> ModelInDB:
        query = """--sql
        INSERT
            INTO
            public.models
        ("path", created_at, updated_at, x_min_max_scaler, y_min_max_scaler, neighborhood_encoder, one_hot_encoder, mse, name, status, gwo_params, epochs, population_size)
        VALUES(%(path)s, NOW(), NOW(), %(x_min_max_scaler)s, %(y_min_max_scaler)s, %(neighborhood_encoder)s, %(one_hot_encoder)s, %(mse)s, %(name)s, %(status)s, %(gwo_params)s, %(epochs)s, %(population_size)s)
        RETURNING id, "path", x_min_max_scaler, y_min_max_scaler, neighborhood_encoder, one_hot_encoder, mse, created_at, updated_at, name, status, gwo_params, epochs, population_size;
        """
        try:
            result = self.conn.fetch_with_retry(sql_statement=query, values={
                "path": model.path,
                "x_min_max_scaler": model.x_min_max,
                "y_min_max_scaler": model.y_min_max,
                "neighborhood_encoder": model.neighborhood_encoder,
                "one_hot_encoder": model.one_hot_encoder,
                "mse": model.mse,
                "name": model.name,
                "status": model.status,
                "gwo_params": json.dumps(model.gwo_params),
                "epochs": model.epochs,
                "population_size": model.population_size
            })
            self.conn.commit()

            if result:
                return ModelInDB(**result)

        except Exception as error:
            _logger.error(f"Error: {str(error)}")

    async def update_status(self, new_status: ModelStatus, model_id: int) -> bool:
        query = """--sql
        UPDATE
            public.models
        SET
            updated_at = NOW(),
            status = %(new_status)s
        WHERE
            id = %(model_id)s
        RETURNING id;
        """
        try:
            result = self.conn.fetch_with_retry(sql_statement=query, values={
                "new_status": new_status,
                "model_id": model_id
            })
            self.conn.commit()

            return bool(result)

        except Exception as error:
            _logger.error(f"Error: {str(error)}")

    async def update(self, model_in_db: ModelInDB) -> bool:
        query = """--sql
        UPDATE
            public.models
        SET
            "path" = %(path)s,
            updated_at = NOW(),
            x_min_max_scaler = %(x_min_max_scaler)s,
            neighborhood_encoder = %(neighborhood_encoder)s,
            one_hot_encoder = %(one_hot_encoder)s,
            mse = %(mse)s,
            y_min_max_scaler = %(y_min_max_scaler)s,
            gwo_params = %(gwo_params)s
        WHERE
            id = %(id)s
        RETURNING id;
        """
        try:
            result = self.conn.fetch_with_retry(sql_statement=query, values={
                "path": model_in_db.path,
                "x_min_max_scaler": model_in_db.x_min_max,
                "y_min_max_scaler": model_in_db.y_min_max,
                "neighborhood_encoder": model_in_db.neighborhood_encoder,
                "one_hot_encoder": model_in_db.one_hot_encoder,
                "mse": model_in_db.mse,
                "id": model_in_db.id,
                "gwo_params": json.dumps(model_in_db.gwo_params)
            })
            self.conn.commit()

            return bool(result)

        except Exception as error:
            _logger.error(f"Error: {str(error)}")

    async def select_latest(self) -> ModelInDB:
        query = """--sql
        SELECT
            id,
            "path",
            x_min_max_scaler AS x_min_max,
            y_min_max_scaler AS y_min_max,
            neighborhood_encoder,
            one_hot_encoder,
            mse,
            created_at,
            updated_at,
            name,
            status,
            gwo_params,
            epochs,
            population_size
        FROM
            public.models m
        ORDER BY
            created_at DESC
        LIMIT 1 OFFSET 0;
        """
        try:
            result = self.conn.fetch_with_retry(sql_statement=query)

            if result:
                return ModelInDB(**result)

        except Exception as error:
            _logger.error(f"Error: {str(error)}")

    async def select_models(self, page: int, page_size: int) -> List[ModelInDB]:
        query = """--sql
        SELECT
            id,
            "path",
            x_min_max_scaler AS x_min_max,
            y_min_max_scaler AS y_min_max,
            neighborhood_encoder,
            one_hot_encoder,
            mse,
            created_at,
            updated_at,
            name,
            status,
            gwo_params,
            epochs,
            population_size
        FROM
            public.models m
        ORDER BY
            created_at DESC
        LIMIT %(page_size)s OFFSET %(page)s;
        """
        try:
            models = []

            results = self.conn.fetch_with_retry(sql_statement=query, values={"page": page - 1, "page_size": page_size}, all=True)

            if results:
                for result in results:
                    models.append(ModelInDB(**result))
     
            return models

        except Exception as error:
            _logger.error(f"Error: {str(error)}")
            return []

    async def select_by_id(self, id: int) -> SummarizedModel:
        query = """--sql
        SELECT
            id,
            name,
            status,
            created_at,
            updated_at
        FROM
            public.models m
        WHERE
            m.id = %(id)s;
        """
        try:
            result = self.conn.fetch_with_retry(sql_statement=query, values={"id": id})

            if result:
                return SummarizedModel(**result)

        except Exception as error:
            _logger.error(f"Error: {str(error)}")

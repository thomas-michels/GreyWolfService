import json
import numpy as np
from typing import List
from app.core.db import DBConnection
from app.core.db.repositories.base_repository import Repository
from app.core.entities import Model, ModelInDB, ModelStatus, SummarizedModel
from app.core.configs import get_logger

_logger = get_logger(__name__)


class ModelRepository(Repository):
    def __init__(self, connection: DBConnection) -> None:
        super().__init__(connection)

    def create(self, model: Model) -> ModelInDB:
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

    def update_status(self, new_status: ModelStatus, model_id: int) -> bool:
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

    def update(self, model_in_db: ModelInDB) -> bool:
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

    def select_latest(self) -> ModelInDB:
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
        WHERE
            m.status = 'READY'
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

    def select_models(self, page: int, page_size: int) -> List[ModelInDB]:
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
        WHERE m.status = 'READY' and m.mse > 0
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

    def select_by_id(self, id: int) -> SummarizedModel:
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

    def select_complete_by_id(self, id: int) -> ModelInDB:
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
        WHERE
            m.status = 'READY' AND m.id = %(model_id)s
        ORDER BY
            created_at DESC
        LIMIT 1 OFFSET 0;
        """
        try:
            result = self.conn.fetch_with_retry(sql_statement=query, values={"model_id": id})

            if result:
                return ModelInDB(**result)

        except Exception as error:
            _logger.error(f"Error: {str(error)}")

    def select_remaining_time(self, model_id: int) -> float:
        query = """--sql
        WITH TimeDiffCTE AS (
            SELECT
                m.id,
                m.gwo_params,
                m.population_size,
                m.epochs,
                mh.epoch AS mh_epoch,
                mh.created_at,
                LAG(mh.created_at) OVER (PARTITION BY m.id ORDER BY mh.epoch) AS prev_created_at
            FROM
                public.models m
            INNER JOIN public.model_histories mh ON
                m.id = mh.model_id
            WHERE m.id = %(model_id)s
        )
        SELECT
            id,
            gwo_params,
            population_size,
            epochs,
            mh_epoch,
            created_at,
            COALESCE(AVG(EXTRACT(EPOCH FROM (created_at - prev_created_at))), 0) AS avg_time_diff
        FROM
            TimeDiffCTE
        GROUP BY
            id, gwo_params, population_size, epochs, mh_epoch, created_at
        ORDER BY
            mh_epoch;
        """
        try:
            results = self.conn.fetch_with_retry(sql_statement=query, values={"model_id": model_id}, all=True)

            times = []

            if not results:
                return 0
            
            last_epoch = results[-1]["mh_epoch"]

            epochs = results[0]["epochs"] + 1
            population_size = results[0]["population_size"]

            for result in results:
                times.append(result["avg_time_diff"])

            mean_time = np.mean(times)

            total_epochs = (epochs * population_size) + 1
            atual_epochs = total_epochs - last_epoch

            return round((atual_epochs * mean_time), 2)

        except Exception as error:
            _logger.error(f"Error: {str(error)}")

    def select_model_statistics(self) -> dict:
        query = """--sql
        SELECT
            m.status,
            count(m.id)
        FROM
            public.models m
        GROUP BY m.status;
        """

        try:
            results = self.conn.fetch_with_retry(sql_statement=query, values={}, all=True)

            status = {}
            parser = {
                "SCHEDULED": "Agendado",
                "TRAINING": "Em treinamento",
                "READY": "Pronto",
                "ERROR": "Error ao treinar"
            }

            if results:
                for result in results:
                    label = result.get("status")
                    if label:
                        status[parser[label]] = result["count"]

            return status
        
        except Exception as error:
            _logger.error(f"Error on select_model_statistics: {str(error)}")
            return {}

    def delete_by_id(self, id: int) -> bool:
        query = """--sql
        DELETE
        FROM
            public.models m
        WHERE
            m.id = %(id)s
        RETURNING 1;
        """

        try:
            result = self.conn.fetch_with_retry(sql_statement=query, values={"id": id})
            self.conn.commit()
            return bool(result)

        except Exception as error:
            _logger.error(f"Error on delete model: {str(error)}")
            return False

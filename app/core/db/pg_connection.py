import psycopg
from psycopg.rows import dict_row
from app.core.configs import get_environment, get_logger
import time

_env = get_environment()
_logger = get_logger(__name__)


class PGConnection:

    def __init__(self) -> None:
        self.__start_connection()

    def execute(self, sql_statement: str, values: tuple = None):
        sql = sql_statement.replace("public", _env.ENVIRONMENT)
        self.cursor.execute(sql, values)

    def commit(self):
        if self.conn:
            self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def fetch(self, all=False):
        return self.cursor.fetchall() if all else self.cursor.fetchone()
    
    def fetch_with_retry(self, sql_statement: str, values: tuple = None, all: bool = False):
        response = None

        for i in range(10):
            try:
                if not self.conn:
                    self.__start_connection()

                self.execute(sql_statement=sql_statement, values=values)
                response = self.fetch(all=all)

                if response:
                    break

            except Exception as error:
                if i >= 9:
                    _logger.error(f"Error: {str(error)}")

                _logger.warning(f"DB retry activated - Count: {i}")
                self.close()
                self.conn = None
                time.sleep(2)

        return response

    def close(self):
        try:
            self.conn.close()

        except Exception:
            ...

    def __start_connection(self):
        try:
            self.conn = psycopg.connect(
                conninfo=(
                    f"host={_env.DATABASE_HOST} "
                    f"port={_env.DATABASE_PORT} "
                    f"user={_env.DATABASE_USER} "
                    f"password={_env.DATABASE_PASSWORD} "
                    f"dbname={_env.DATABASE_NAME} "
                    f"keepalives_idle=5 "
                    f"keepalives_interval=2 "
                    f"keepalives_count=2"
                ),
                autocommit=False,
                row_factory=dict_row
            )
            self.cursor = self.conn.cursor()

        except Exception:
            self.conn = None
            self.cursor = None

    def __enter__(self):
        self.__start_connection()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

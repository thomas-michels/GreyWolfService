import psycopg
from psycopg.rows import dict_row
from app.core.configs import get_environment

from .base_connection import DBConnection

_env = get_environment()


class PGConnection(DBConnection):

    def __init__(self) -> None:
        self.__start_connection()

    def execute(self, sql_statement: str, values: tuple = None):
        sql = sql_statement.replace("public", _env.ENVIRONMENT)
        self.cursor.execute(sql, values)

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def fetch(self, all=False):
        return self.cursor.fetchall() if all else self.cursor.fetchone()

    def close(self):
        self.conn.close()

    def __start_connection(self):
        self.conn = psycopg.connect(
            conninfo=(
                f"host={_env.DATABASE_HOST} "
                f"port={_env.DATABASE_PORT} "
                f"user={_env.DATABASE_USER} "
                f"password={_env.DATABASE_PASSWORD} "
                f"dbname={_env.DATABASE_NAME}"),
            autocommit=False,
            row_factory=dict_row,
            application_name=_env.APPLICATION_NAME
        )
        self.cursor = self.conn.cursor()

    def __enter__(self):
        self.__start_connection()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

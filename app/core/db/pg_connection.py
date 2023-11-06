import psycopg
from psycopg.rows import dict_row
from app.core.configs import get_environment
import time

_env = get_environment()


class PGConnection:

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
    
    def fetch_with_retry(self, sql_statement: str, values: tuple = None, all: bool = False):
        response = None

        for i in range(5):
            try:
                if not self.conn:
                    self.__start_connection()

                self.execute(sql_statement=sql_statement, values=values)
                response = self.fetch(all=all)

                if response:
                    break

                time.sleep(2)

            except Exception:
                ...

        return response

    def close(self):
        self.conn.close()

    def __start_connection(self):
        try:
            self.conn = psycopg.connect(
                conninfo=(
                    f"host={_env.DATABASE_HOST} "
                    f"port={_env.DATABASE_PORT} "
                    f"user={_env.DATABASE_USER} "
                    f"password={_env.DATABASE_PASSWORD} "
                    f"dbname={_env.DATABASE_NAME}"),
                autocommit=False,
                row_factory=dict_row
            )
            self.cursor = self.conn.cursor()

        except Exception:
            self.conn = None

    def __enter__(self):
        self.__start_connection()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

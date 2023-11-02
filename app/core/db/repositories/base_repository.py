from app.core.db import DBConnection


class Repository:

    def __init__(self, connection: DBConnection) -> None:
        self._conn = connection

    @property
    def conn(self) -> DBConnection:
        return self._conn

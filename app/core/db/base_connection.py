from abc import ABC, abstractmethod


class DBConnection(ABC):

    @abstractmethod
    def execute(self, sql_statement: str, values: tuple = None):
        """
        Method to execute query
        """

    @abstractmethod
    def commit(self):
        """
        Method to commit current changes
        """

    @abstractmethod
    def rollback(self):
        """
        Method to rollback current changes
        """

    @abstractmethod
    def fetch(self, all=False):
        """
        Method to get all or one occurence from database
        """

    @abstractmethod
    def fetch_with_retry(self, sql_statement: str, values: tuple = None, all=False):
        """
        Method to get all or one occurence from database
        """

    @abstractmethod
    def close(self):
        """
        Method to close connection
        """

    @abstractmethod
    def __enter__(self):
        """
        Method to start connection
        """

    @abstractmethod
    def __exit__(self):
        """
        Method to close connection
        """

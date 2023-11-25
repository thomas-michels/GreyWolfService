"""
File to queue callback
"""

from typing import Any
from kombu import Queue
from app.core.configs import get_logger

_logger = get_logger(name=__name__)


class QueueCallback:
    """
    This class save queue and callback function
    """

    __queue_name: str
    __queue: Queue
    __function: Any

    def __init__(self, queue_name: str, queue: Queue, function: Any) -> None:
        self.__queue_name = queue_name
        self.__queue = queue
        self.__set_function(function)

    def get_function(self) -> Any:
        """
        Method to return function

        :return:
            Any
        """
        return self.__function

    def get_queue(self) -> Queue:
        """
        Method to return queue

        :return:
            Queue
        """
        return self.__queue

    def get_queue_name(self) -> str:
        """
        Method to return queue name

        :return:
            str
        """
        return self.__queue_name

    def __set_function(self, function) -> None:
        """
        Method to set function of callback

        :param:
            function: Any

        :return:
            NoReturn
        """
        try:
            self.__function = function

        except AttributeError as error:
            _logger.error(f"Error on set function in QueueCallback - {error}")
            raise Exception("CallbackNotMethod")

        except KeyError as error:
            _logger.error(f"Error on set function in QueueCallback - {error}")
            raise Exception("FunctionAnnotation")

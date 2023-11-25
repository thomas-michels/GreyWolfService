"""
Queue manager file
"""

from typing import List
from kombu import Queue
from app.core.configs import get_environment, get_logger
from app.worker.consumer.queues.queue_callback import QueueCallback
from app.worker.consumer.queues import create_queue
from app.callbacks.callback_interface import Callback

_env = get_environment()
_logger = get_logger(name=__name__)


class QueueManager():
    """
    This class create and save queues in list
    """

    def __init__(self) -> None:
        super().__init__()
        self._queues: List[QueueCallback] = []

    def destroy(self):
        self._queues = []

    def register_callback(self, queue_name: str, callback: Callback) -> None:
        """
        Method to register new callback

        :parms:
            queue_name: str
            callback: Callback

        :return:
            NoReturn
        """
        try:

            queue = create_queue(queue_name.upper(), _env.RBMQ_EXCHANGE)

            queue_callback = QueueCallback(
                queue_name=queue_name.upper(), queue=queue, function=callback
            )

            if self.get_queue_by_name(queue_callback.get_queue_name()):
                raise Exception("CallbackAlreadyCreated")

            self._queues.append(queue_callback)
            _logger.info(f"QUEUE: {queue_name.upper()} REGISTRED")

        except Exception as error:
            _logger.error(f"Error on register callbacks - Error: {error}")

    def get_queues(self) -> List[Queue]:
        """
        Method to return all queues

        :return:
            List[Queue]
        """
        return [callback.get_queue() for callback in self._queues]

    def get_name_queues(self) -> List[str]:
        """
        Method to return all name of queues

        :return:
            List[str]
        """
        return [callback.get_queue_name() for callback in self._queues]

    def get_function(self, queue_name: str) -> Callback:
        """
        Method to get callback function

        :return:
            Any
        """
        for callback in self._queues:
            if callback.get_queue_name() == queue_name.upper():
                return callback.get_function()

        raise Exception("QueueNotFound")

    def get_queue_by_name(self, queue_name: str) -> Queue:
        """
        Method to get queue by name

        :return:
            Any
        """
        for callback in self._queues:
            if callback.get_queue_name() == queue_name.upper():
                return callback.get_queue()

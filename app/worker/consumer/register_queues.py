"""
    Module for register queues
"""
from .manager import QueueManager
from app.core.configs import get_logger, get_environment
from app.callbacks.train_model import TrainModelCallback

_logger = get_logger(name=__name__)
_env = get_environment()


class RegisterQueues:
    """
    RegisterQueues class
    """

    @staticmethod
    def register() -> QueueManager:
        _logger.info("Starting QueueManager")
        queue_manager = QueueManager()

        queue_manager.register_callback(
            _env.TRAIN_MODEL_CHANNEL, TrainModelCallback
        )

        _logger.info("All queues started")

        return queue_manager

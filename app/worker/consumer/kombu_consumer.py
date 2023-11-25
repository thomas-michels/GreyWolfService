"""
Kombu worker class module
"""

from app.core.db import PGConnection
from kombu import Connection
from kombu.mixins import ConsumerMixin
from app.core.configs import get_logger, get_environment
from app.worker.consumer.manager import QueueManager
from app.worker.utils.validate_event import payload_conversor
import json

_logger = get_logger(name=__name__)
_env = get_environment()


class KombuWorker(ConsumerMixin):
    """
    This class is Kombu Worker
    """

    def __init__(self, connection: Connection, queues: QueueManager):
        self.queues = queues
        self.connection = connection

    def get_consumers(self, Consumer, channel):
        return [
            Consumer(
                queues=self.queues.get_queues(),
                callbacks=[self.process_task],
                prefetch_count=_env.PREFETCH_VALUE
            )
        ]

    def process_task(self, body, message):
        try:
            infos = message.delivery_info
            _logger.info(f"Message received at {infos['routing_key']}")
            message.ack()
            callback = self.queues.get_function(infos["routing_key"])
            pg_connection = PGConnection()
            callback = callback(pg_connection)
            event_schema = payload_conversor(body)
            if event_schema:
                if isinstance(event_schema.payload, str):
                    event_schema.payload = json.loads(event_schema.payload)

                callback.handle(event_schema)

            _logger.info(f"Message consumed at {event_schema.id}")

        except Exception as error:
            _logger.error(f"Error on process_task - {error}")

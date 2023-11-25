"""
    Module for Kombu Producer class
"""

from kombu.mixins import Producer
from app.worker.utils import start_connection_bus, connect_on_exchange
from app.worker.utils.event_schema import EventSchema
from app.core.configs import get_logger, get_environment

_env = get_environment()
_logger = get_logger(name=__name__)


class KombuProducer:
    """
    Class for Producer to send messages in queues
    """

    @classmethod
    def send_messages(cls, message: EventSchema) -> bool:
        """
        Method to send messages

        :param message: EventSchema

        :return: bool
        """
        try:
            with start_connection_bus() as bus_conn:
                producer = Producer(bus_conn)
                producer.publish(
                    body=message.model_dump(),
                    exchange=connect_on_exchange(_env.RBMQ_EXCHANGE),
                    routing_key=message.sent_to,
                )

            _logger.info(f"Sent message to {message.sent_to}")
            return True

        except Exception as error:
            _logger.error(f"Error on send message to {message.sent_to}. Error: {error}")

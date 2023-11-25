"""
    Module for create queues
"""
from kombu import Queue
from app.worker.utils import connect_on_exchange


def create_queue(queue_name: str, exchange_name: str) -> Queue:
    """
    Function to create queue on rabbitMQ

    :param queue_name: str
    :param exchange_name: str

    :return: Queue
    """
    return Queue(
        name=queue_name,
        exchange=connect_on_exchange(exchange_name),
        routing_key=queue_name,
        queue_arguments={
            "x-dead-letter-exchange": queue_name,
            "x-dead-letter-routing-key": "delay",
            "durable": True,
        },
        auto_delete=False,
    )

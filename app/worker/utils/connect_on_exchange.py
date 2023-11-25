"""
    Module for exchange connect
"""

from kombu import Exchange


def connect_on_exchange(exchange_name: str) -> Exchange:
    """
    Function to connect on rabbitMQ Exchange

    :param exchange_name: str

    :return: Exchange
    """
    return Exchange(name=exchange_name, type="direct")

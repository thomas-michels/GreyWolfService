"""
    Module for start connection with RabbitMQ
"""
from kombu import Connection
from app.core.configs import get_environment

_env = get_environment()


def start_connection_bus() -> Connection:
    return Connection(
        hostname=_env.RBMQ_HOST,
        userid=_env.RBMQ_USER,
        password=_env.RBMQ_PASS,
        port=_env.RBMQ_PORT,
        virtual_host=_env.RBMQ_VHOST
    )

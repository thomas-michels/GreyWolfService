from app.core.configs import get_logger
from app.worker import (
    KombuWorker,
    RegisterQueues,
    start_connection_bus
)
import signal

_logger = get_logger(__name__)


class Consumer:

    def __init__(self) -> None:
        signal.signal(signal.SIGTERM, self.terminate)
        signal.signal(signal.SIGINT, self.terminate)

    def start(self):
        try:
            queues = RegisterQueues.register()

            _logger.info("Starting Worker")

            with start_connection_bus() as conn:
                worker = KombuWorker(conn, queues)
                worker.run()

        except KeyboardInterrupt:
            _logger.info("Stopping Worker")
            quit()

    def terminate(self, *args):
        quit()

import logging
import time
from logging.handlers import RotatingFileHandler


def create_rotating_log(path):
    """
    Creates a rotating log
    """
    logger = logging.getLogger("Rotating Log")
    logger.setLevel(logging.DEBUG)

    handler = RotatingFileHandler(path, maxBytes=20, backupCount=5)
    logger.addHandler(handler)

    for i in range(10):
        logger.debug(f"This is test log line {i}")
        time.sleep(1.5)

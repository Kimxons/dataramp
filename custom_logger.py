import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOGGING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dh_logs")

if not os.path.exists(LOGGING_DIR):
    os.makedirs(LOGGING_DIR)

LOG_FORMAT = "%(asctime)s | %(levelname)s : %(message)s"


class Logger:
    def __init__(self, logger_name=None, filename=None, log_level=logging.DEBUG):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        formatter = logging.Formatter(LOG_FORMAT)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        if filename is not None:
            log_file = os.path.join(LOGGING_DIR, filename)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=1024 * 1024, backupCount=5
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def close(self):
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)

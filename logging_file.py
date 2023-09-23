import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOGGING_DIR = "dh_logs"

if not os.path.exists(LOGGING_DIR):
    os.makedirs(LOGGING_DIR)


class Logger:
    def __init__(self, logger_name=None, filename=None, log_level=logging.DEBUG):  # noqa: E501
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False

        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s : %(message)s')

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if filename is not None:
            log_file = os.path.join(LOGGING_DIR, filename)
            fhandler = RotatingFileHandler(
                log_file, maxBytes=1024*1024, backupCount=5)  # 1MB
            fhandler.setFormatter(formatter)
            self.logger.addHandler(fhandler)

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

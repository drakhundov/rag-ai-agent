"""Used to handle log/system level error handling."""
import logging


def log_and_raise(logger: logging.Logger, err: Exception):
    logger.error(str(err), exc_info=err)
    raise err

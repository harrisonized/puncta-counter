"""Module to handle error logging
Adapted from harrison_functions: https://github.com/harrisonized/harrison-functions
"""

import re
import logging
import datetime as dt


class LogFilter(logging.Filter):
    """flatten messages (remove embedded new line, etc)
    """
    regexp = re.compile('\\r|\\n')
    substitute = ' \\\\n '

    def filter(self, record):
        # print('FILTER: {}'.format(record.msg))
        record.msg = LogFilter.regexp.sub(LogFilter.substitute, record.msg)
        return True


def configure_handler(handler, level, log_filter=None):
    """Add formatter and filter to logger or handler
    """

    formatter = logging.Formatter(
        '%(asctime)s\t%(process)d\t'
        '%(module)s:%(funcName)s:%(lineno)s\t%(levelname)s\t'
        '%(message)s'
    )

    handler.setLevel(level)
    if "setFormatter" in dir(handler):
        handler.setFormatter(formatter)
    if log_filter:
        handler.addFilter(log_filter)

    return handler


def configure_logger(logger, level, filename):
    # configure logging at logger level!

    # base logger
    log_filter = LogFilter()
    logger = configure_handler(logger, level, log_filter=log_filter)

    # file handler
    now = dt.datetime.now().strftime('%Y%m%d_%H%M%S')  # don't need %f
    file_handler = logging.FileHandler(f"{filename}-{now}.log")
    file_handler = configure_handler(file_handler, level, log_filter)
    logger.addHandler(file_handler)

    # stream handler
    stream_handler = logging.StreamHandler()
    stream_handler = configure_handler(stream_handler, level, log_filter)
    logger.addHandler(stream_handler)

    return logger

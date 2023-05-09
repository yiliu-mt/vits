import json
import logging.config
import sys
from datetime import datetime
from pytz import timezone
from logging import Formatter, StreamHandler
from logging.handlers import RotatingFileHandler
import pytz

def timetz(*args):
    tz = timezone('Asia/Shanghai') # UTC, Asia/Shanghai, Europe/Berlin
    return datetime.now(tz).timetuple()


def create_logger_from_conf(log_conf_path="tools/conf/logging_conf.json", log_file_path=None):
    # read config
    with open(log_conf_path) as f:
        config = json.load(f)
    if log_file_path is None:
        log_file_path = config.get("log_filename", "service_management.log")
    log_keep_size = config.get("log_keep_size", 20 * 1024 * 1024)
    log_backup_count = config.get("log_backup_count", 10)
    log_format = config.get("log_format", "%(asctime)s | %(module)s | %(thread)d | %(levelname)s : %(message)s")
    log_date_fmt = config.get("log_date_fmt", "%Y-%m-%d %H:%M:%S")

    logging.basicConfig(format="%(asctime)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    logging.Formatter.converter = timetz
    logger = logging.getLogger(__name__)

    # set logger
    logging.captureWarnings(True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = Formatter(fmt=log_format, datefmt=log_date_fmt)
    rotating_file_handler = RotatingFileHandler(filename=log_file_path, maxBytes=log_keep_size,
                                                backupCount=log_backup_count)
    rotating_file_handler.setFormatter(fmt)
    rotating_file_handler.setLevel(logging.INFO)

    console_handler = StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(rotating_file_handler)
    logger.addHandler(console_handler)
    return logger

LOGGER = create_logger_from_conf()

def get_current_time():
    '''Return the current time: 2022-09-22 13:45:59'''
    return datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    LOGGER.debug('debug')
    LOGGER.info('info')
    LOGGER.warning('warn')
    LOGGER.error('error')
    LOGGER.critical('critical')

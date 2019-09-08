import os
import yaml
import logging
from logging import Formatter
from logging.handlers import WatchedFileHandler

phase = 'development'


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))

bigquery_config = yaml.load(
    open(
        os.path.join(root_path, 'config', 'bigquery.yaml'),
        'r'
    )
)


def get_logger(name, **kwargs):
    logger = logging.getLogger(name)
    if kwargs.get('logging_mode'):
        logger.setLevel(logging_mode_converter(kwargs.get('logging_mode')))
    if not os.path.isdir('log'):
        os.mkdir('log')
    file_path = f'log/{name}.log'
    file_handler = WatchedFileHandler(
        file_path,
        encoding='utf-8'
    )
    file_handler.setFormatter(
        Formatter('[%(levelname)s]|%(asctime)s|%(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)
    return logger


def logging_mode_converter(mode):
    if mode == "DEBUG":
        return logging.DEBUG
    elif mode == "INFO":
        return logging.INFO
    else:
        return logging.DEBUG


__all__ = ['phase', 'root_path', 'bigquery_config']

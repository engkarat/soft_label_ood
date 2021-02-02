import os
import logging
import sys


class Logger(object):

    _logger = None

    def __init__(self, name, log_dir=None, no_std_out=False, log_file_name='run'):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not logger.handlers:
            # usually keep the LOGGING_DIR defined in some global settings file
            if not log_dir:
                path = os.path.join('log', '%s' % name)
            else:
                path = log_dir

            if not os.path.isdir(path):
                os.makedirs(path)

            file_name = os.path.join(path, '{}.log'.format(log_file_name))

            handler = logging.FileHandler(file_name)
            formatter = logging.Formatter(
                '%(asctime)s %(levelname)s - [%(name)s]: %(message)s'
            )
            handler.setFormatter(formatter)
            handler.setLevel(logging.DEBUG)
            logger.addHandler(handler)
            # for ease of use, also add a handler that outputs to the screen
            if not no_std_out:
                ch = logging.StreamHandler(sys.stdout)
                ch.setLevel(logging.DEBUG)
                ch.setFormatter(formatter)
                logger.addHandler(ch)
        self._logger = logger

    def get(self):
        """
        :return: returns the singleton logger
        """
        return self._logger
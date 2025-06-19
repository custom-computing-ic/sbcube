import os
from colorama import Fore, Style

class Logger:
    def __init__(self, log_level, **kwargs):

        import logging

        class ColorFormatter(logging.Formatter):
            LEVEL_COLORS = {
                'DEBUG': Fore.MAGENTA,
                'INFO': Fore.CYAN,
                'WARNING': Fore.YELLOW,
                'ERROR': Fore.RED,
                'CRITICAL': Fore.RED + Style.BRIGHT,
            }


            def format(self, record):
                time_str = self.formatTime(record, "%H:%M:%S.%f")[:-3]
                level_color = self.LEVEL_COLORS.get(record.levelname, "")
                level_str = f"{level_color}{record.levelname:<8}{Style.RESET_ALL}"
                return f"{time_str} | {level_str} | {record.getMessage()}"


        options = { }
        if log_level is not None:
            options['level'] = log_level.upper()
        options['format'] = '%(message)s'

        logging.basicConfig(**options)
        logger = logging.getLogger()
        for handler in logger.handlers:
            handler.setFormatter(ColorFormatter())

        self.info = logger.info
        self.debug = logger.debug
        self.warning = logger.warning
        self.error = logger.error


    def artifact(self, **kwargs):
        return
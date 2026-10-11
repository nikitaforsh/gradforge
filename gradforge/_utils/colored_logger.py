import logging

from colorama import Fore, Style, init


class ColoredFormatter(logging.Formatter):

    COLORS = {  
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        color = self.COLORS.get(record.levelno, Fore.WHITE)
 
        fmt = f"{Fore.LIGHTBLACK_EX}%(asctime)s {color}[%(levelname)s]{Style.RESET_ALL} %(message)s"
        
        formatter = logging.Formatter(fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

class Logger:
    def __init__(self):
        init(autoreset=True)
        self.logger = logging.getLogger("AppLogger")
        self.logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter())
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
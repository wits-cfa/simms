import logging
import os
from importlib import metadata

__version__ = metadata.version(__package__)

PCKGDIR = os.path.dirname(os.path.abspath(__file__))

SCHEMADIR = os.path.join(__path__[0], "schemas")


def set_logger(name, level="WARNING"):
    if isinstance(level, str):
        level = getattr(logging, level, 10)
        
        # create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # create formatter
    formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)-8s| %(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    
    return logger



class BinClass:
    def __init__(self):
        self.skysim = "skysim"
        self.telsim = "telsim"


BIN = BinClass()

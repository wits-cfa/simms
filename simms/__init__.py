import os
import logging

SCHEMADIR = os.path.join(__path__[0], "schemas")

def get_logger(name, level="DEBUG"):

    if isinstance(level, str):
        level = getattr(logging, level, 10)

    format_string = '%(asctime)s-%(name)s-%(levelname)-4s| %(message)s'
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format=format_string,
                        datefmt='%m:%d %H:%M:%S')

    return logging.getLogger(name)

LOG = get_logger("simms")



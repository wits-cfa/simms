class ValidationError(Exception):
    pass


class CatalogueError(Exception):
    pass


class ParameterError(Exception):
    pass


def isnummber(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
# nicked from https://www.hackertouch.com/how-to-get-a-list-of-class-attributes-in-python.html
def get_class_attributes(cls):
    return [item for item in cls.__dict__ if not callable(getattr(cls, item)) and not item.startswith('__')]


class ObjDict(object):
    def __init__(self, items):
        """
        Converts a dictionary into an object. 

        """
        # First give this objects all the attributes of the input dicttionary
        for item in dir(dict):
            if not item.startswith("__"):
                setattr(self, item, getattr(items, item, None))
        # Now set the dictionary values as attributes
        self.__dict__.update(items)

class ValidationError(Exception):
    pass


class ASCIISourceError(Exception):
    pass


class FITSSkymodelError(Exception):
    pass


class ParameterError(Exception):
    pass

class ASCIISkymodelError(Exception):
    pass

class SkymodelSchemaError(AttributeError):
    pass

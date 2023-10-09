from  simms.utilities import ValidationError, ListSpec, BASE_TYPES
from copy import deepcopy
from simms import LOG


class Parameter(object):
    def __init__(self, key, dtype, info, default=None, required=False):
        self.key = key 
        self.dtype = dtype
        self.info = info
        # ensure that the default is not overwritten if value is updated
        self.default = deepcopy(default)
        self.value = self.default
        self.required = required
        self.islist = False

        if isinstance(self.dtype, str):
            if self.dtype.startswith("List["):
                self.dtype = ListSpec(self.dtype)
                self.islist = True
            elif self.dtype in BASE_TYPES.keys():
                self.dtype = getattr(BASE_TYPES, self.dtype)
            else:
                raise ValidationError(f"Type {self.dtype} is not supported.")

        if not isinstance(required, bool):
            raise ValidationError("The required option has to be a boolean")
    
    def validate_value(self, value=None):
        """
        Extends the isinstance() function to handle our MyList types

        Parameters
        ---------
        value: Any
            The value being checked
        dtype:
            The type that value has to be

        
        Returns
        --------------
        Boolean value indicating whether value is of type dtype.
        """
        value = value or self.value
        if self.islist:
            if not isinstance(value, list):
                LOG.warning(f"List parameter, {self.key}, given as a single value."
                            f" Setting it to a single-valued List")
                value = [value]
            self.dtype.set_dtype()
            return all( isinstance(item, self.dtype.dtype) for item in value )
        else:
            return isinstance(value, self.dtype)
    
    def update_value(self, value):
        if self.validate_value(value):
            self.value = value
        else:
            raise ValidationError( f"Parameter has wrong type. "
                                  f"Expected type: {self.dtype}, "
                                  f"Actual type: {type(value)}, "
                                  f"Actual value: {value}")
    
def validate(valme, schemafile=None):

    valme.set_schema(schemafile)
    section = valme.schema[valme.schema_section]
    valme.validate_section(valme.schema_section)

    for key in section:
        value = getattr(valme, key, None)
        # valid parameters can only be dicts, strings are either meta or infomation data
        # which does need to be validated

        if isinstance(section[key], str):
            continue 

        param = Parameter(key, **section[key])
        if value is None:
            if param.required:
                raise ValidationError(f"Required parameter '{key}' has not been set")
        else:
            param.update_value(value)

        LOG.debug(f"Setting key: {param.key}, value: {param.value}")

from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union
from enum import Enum
import re

class ValidationError(Exception):
    pass

@dataclass
class MyList (list):
    dtype: str
    def __call__(self):
        if isinstance(self.dtype,str):
            thematch = re.findall("^List*\[([a-zA-Z{1,}].*\w{1,})\]", self.dtype)
            if thematch:
                self.thematch = thematch[0]
        if not hasattr(self, 'thematch'):
            raise ValidationError(f"Type {self.dtype} is not supported.")
    



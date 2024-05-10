import os
from scabha import configuratt
from scabha.cargo import Parameter, EmptyDictDefault
from omegaconf import OmegaConf
from typing import Dict, Any, List
from dataclasses import dataclass
from tests.utilities import File

thisdir = os.path.dirname(__file__)

@dataclass
class SchemaSpec:
    inputs: Dict[str,Parameter]
    outputs: Dict[str,Parameter]
    libs: Dict[str, Any] = EmptyDictDefault()

def load(name: str, use_sources: List = []) -> Dict:
    """Load a scabha-style parameter defintion using.

    Args:
        name (str): Name of parameter definition to load
        use_sources (List, optional): Parameter definition dependencies 
        (a.k.a files specified via_include)

    Returns:
        Dict: Schema object
    """
    
    parser = os.path.join(thisdir, f'{name}.yaml')
    args_defn = OmegaConf.structured(SchemaSpec)
    struct_args, _ = configuratt.load_nested([parser], structured=args_defn,
                                             use_sources=use_sources, use_cache=False)
    schema = OmegaConf.create(struct_args)
    
    return schema[name]

def load_sources(sources: List[str|File]):
    __sources = [None]*len(sources)
    for i, src in enumerate(sources):
        if isinstance(src, str):
            if src.endswith((".yaml", ".yml")):
                sources[i] = File(src)
            else:
                try:
                    sources[i] = File(os.path.join(thisdir,f"{src}.yaml"))
                except FileNotFoundError:
                    raise FileNotFoundError(f"Name {src} does not match a known parameter file.")
                
        __sources[i], _ = configuratt.load(sources[i], use_cache=False)
    return __sources[i]

                
import os
from simms.parser_config.utils import load
from scabha.schema_utils import clickify_parameters
from scabha import configuratt
import click
from omegaconf import OmegaConf



thisdir = os.path.dirname(__file__)

command = "observe"

__sources = ["skysim.yaml", "telescope.yaml"]
sources = list(map(lambda x: configuratt.load(x)[0], 
                   [f"{os.path.join(thisdir, src)}" for src in __sources ]
))

config = load(command, use_sources=sources)


@click.command(command)
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    print(opts)
    
runit()

  
import simms
from simms.parser_config.utils import load
from scabha.schema_utils import clickify_parameters
import click
from omegaconf import OmegaConf

command = "telescope"
config = load(command)


@click.command(command)
@click.version_option(str(simms.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    print(opts)
    
runit()
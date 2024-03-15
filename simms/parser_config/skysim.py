import simms
from simms.parser_config.utils import load, load_sources
from scabha.schema_utils import clickify_parameters
import click
from omegaconf import OmegaConf

command = "skysim"
sources = load_sources(["library/sys_noise"])
config = load(command, use_sources=sources)


@click.command(command)
@click.version_option(str(simms.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    print(opts)
    
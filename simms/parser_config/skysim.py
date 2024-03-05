import simms
from simms.parser_config.utils import load, load_sources
from scabha.schema_utils import clickify_parameters, paramfile_loader
import click
from omegaconf import OmegaConf
from simms import BIN

log = init_logger(BIN.skysim)

command = BIN.im_plane
sources = load_sources(["library/sys_noise"])
thisdir  = os.path.dirname(__file__)#how does the code know what the __file__ means

source_files = glob.glob(f"{thisdir}/library/*.yaml") #this gets all the files in the library that end in yamls
sources = [File(item) for item in source_files] #converts all yaml files found above to dtype file
parserfile = File(f"{thisdir}/{command}.yaml") #finds the paserfile which sould be in this directory (and match commandname)
config = paramfile_loader(parserfile, sources)[command] #what does this line do?


@click.command(command)
@click.version_option(str(simms.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    print(opts)

    inpms = File(opts.ms)
    sourcecat = File(opts.source_catalogue)
    die = File(opts.die)
    dde = File(opts.dde) #for these that are not required is this still valid 
    #or should it be:
    if opts.die.EXISTS:
        die = File(opts.die)
    sourcetype = opts.source_type #will this now just be the string of source type. 
    spectype = opts.spectrum

    


    
runit(
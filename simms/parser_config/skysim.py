import os
import os
import simms
from simms.parser_config.utils import load, load_sources, thisdir as skysimdir
from scabha.schema_utils import clickify_parameters, paramfile_loader
from scabha.basetypes import File
import click
from omegaconf import OmegaConf
from simms import BIN, get_logger 
#from simms.skymodel import catalogue, thisdir as skysimdir
import glob
from simms.utilities import CatalogueError

log = get_logger(BIN.skysim)

command = BIN.skysim
command = BIN.skysim
sources = load_sources(["library/sys_noise"])
thisdir  = os.path.dirname(__file__)
thisdir  = os.path.dirname(__file__)
config = load(command, use_sources=sources)

source_files = glob.glob(f"{thisdir}/library/*.yaml") 
sources = [File(item) for item in source_files] 
parserfile = File(f"{thisdir}/{command}.yaml") 
config = paramfile_loader(parserfile, sources)[command] 


#skyspec = paramfile_loader(os.path.join(skysimdir, "skysim.yaml"))
#print(skyspec)


@click.command(command)
@click.version_option(str(simms.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    cat = opts.catalogue
    ms = opts.ms
    map_path = opts.mapping
    mapdata = OmegaConf.load(map_path)
    mapcols = OmegaConf.create({})
    cattypes = []
    for key in mapdata:
        catkey = mapdata.get(key) or key
        mapcols[key] = (catkey, [])

    print(mapcols)



    with open(cat) as stdr:
        header = stdr.readline().strip()

        if not header.startswith("#format:"):
            raise CatalogueError("Catalogue needs to have a header starting with the string #format:")
        
        header = header.strip().replace("#format:", "").strip()
        header = header.split()

        for line in stdr.readlines():
            if line.startswith("#"):
                continue
            rowdata = line.strip().split()
            assert len(rowdata) == len(header) #raise cat error
            #print(line)
            for key in mapcols:
                #print(key, mapcols[key])
                catkey = mapcols[key][0]
                print(catkey)
                if catkey not in header:
            
                    continue
                
                index = header.index(catkey)
                mapcols[key][1].append(rowdata[index])

        print(mapcols)    #print(mapcols)
            #print(header)
            #print(line[2])
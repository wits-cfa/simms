import simms
from simms.parser_config.utils import load, load_sources
from scabha.schema_utils import clickify_parameters
import click
from omegaconf import OmegaConf
import yaml

command = "skysim"
sources = load_sources(["library/sys_noise"])
config = load(command, use_sources=sources)


@click.command(command)
@click.version_option(str(simms.__version__))
@clickify_parameters(config)
def runit(**kwargs):
    opts = OmegaConf.create(kwargs)
    cat = opts.get('catalogue')
    ms = opts.get('ms')
    map_path = opts.get('mapping')
    #print(map_path)

catcols = []
with open('skyspec.yaml', 'r') as m:
    mapdata = yaml.load(m, Loader=yaml.SafeLoader)
    for key in mapdata:
        value = mapdata.get(key)
        if value != None:
            key = value
        catcols.append(key)
            
#print(catcols)

#with open("testsky.txt", 'r') as cat_file:
    # Read the header line to get column names
    #header = cat_file.readline().strip().split()
    #for cols in catcols:
        #print(cols)
       # print(header)

class Source:
    def __init__(self, **kwargs):
        for column_name in catcols:
            value = kwargs.get(column_name, None)
            if isinstance(value, str) and value.endswith(("as")):
                value = float(value[:-2]) / 3600
            if isinstance(value, str) and value.endswith(("arcsec")):
                value = float(value[:-6]) / 3600
            if isinstance(value, str) and value.endswith(("GHz")):
                value = float(value[:-3]) * 1e9
            if isinstance(value, str) and value.endswith(("MHz")):
                value = float(value[:-3]) * 1e6
            if isinstance(value, str) and value.endswith(("kHz")):
                value = float(value[:-3]) * 1e3
            setattr(self, column_name, value)
            print(f"Setting {column_name} to {value}")
            


def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def parse_file(filename, columns):
    sources = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        header = lines[0].strip().split()
        for line in lines[1:]:
            data = line.strip().split()
            data_dict = {col: (float(val) if is_numeric(val) and (col.startswith('stokes') or col.startswith('cont_coeff') or col.startswith('line')) else val)
                         for col, val in zip(columns, data)}
            source = Source(**data_dict)

    return sources


filename = 'testsky.txt'
sources = parse_file(filename, catcols)

print(Source)


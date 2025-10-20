from omegaconf import OmegaConf
from scabha.basetypes import File
from simms.utilities import (
    CatalogueError,
    isnummber,
    )
from simms.skymodel.source_factory import CatSource

def catalogue2dict(catfile, map_path, delimiter):
    
    mapdata = OmegaConf.load(map_path)
    mapcols = OmegaConf.create({})

    for key in mapdata:
        keymap = mapdata.get(key)
        if keymap:   
            mapcols[key] = (keymap.name or key, [], keymap.get("unit"))
        else:
            mapcols[key] = (key, [], None)
            
    with open(catfile) as stdr:
        header = stdr.readline().strip()

        if not header.startswith("#format:"):
            raise CatalogueError("Catalogue needs to have a header starting with the string #format:")
        
        header = header.strip().replace("#format:", "").strip()
        header = header.split(delimiter)

        for line in stdr.readlines():
            if line.startswith("#"):
                continue
            if not line.strip():
                continue
            rowdata = line.strip().split(delimiter)
            if len(rowdata) != len(header):
                raise CatalogueError("The number of elements in one or more rows does not equal the\
                                    number of expected elements based on the number of elements in the\
                                    header")
            for key in mapcols:
                catkey = mapcols[key][0]
                if catkey not in header:
                    continue
                
                index = header.index(catkey)
                value = rowdata[index]
                if isnummber(value) and mapcols[key][2]:
                    value += mapcols[key][2]
                
                mapcols[key][1].append(value)
    
    # validate mapcols to ensure that the required columns were read successfully
    for col in ["name", "ra", "dec", "stokes_i"]:
        if not mapcols[col][1]: # if the list storing column's data is empty
            if col == "name": # user might not understand what simply "name" means
                raise CatalogueError(
                    f"Failed to identify required column corresponding to source name/ID in the catalogue."
                    " Please ensure that catalogue column headers match those in mapping file."
                )
            else:
                raise CatalogueError(
                f"Failed to identify required column corresponding to '{col}' in the catalogue."
                " Please ensure that catalogue column headers match those in mapping file."
            )
    return mapcols


def dict2sources(data):
    num_sources = len(data['name'][1])
    sources = []
    for i in range(num_sources):
        source = CatSource(
            name=data["name"][1][i],
            
            ra=data["ra"][1][i],
            dec=data["dec"][1][i],

            emaj=data["emaj"][1][i] if i < len(data["emaj"][1]) else None,
            emin=data["emin"][1][i] if i < len(data["emin"][1]) else None,
            pa=data["pa"][1][i] if i < len(data["pa"][1]) else "0deg",

            stokes_i=data["stokes_i"][1][i],
            stokes_q=data["stokes_q"][1][i] if i < len(data["stokes_q"][1]) else None,
            stokes_u=data["stokes_u"][1][i] if i < len(data["stokes_u"][1]) else None,
            stokes_v=data["stokes_v"][1][i] if i < len(data["stokes_v"][1]) else None,

            cont_reffreq=(
                data["cont_reffreq"][1][i] if i < len(data["cont_reffreq"][1]) else None
            ),
            line_peak=(
                data["line_peak"][1][i] if i < len(data["line_peak"][1]) else None
            ),
            line_width=(
                data["line_width"][1][i] if i < len(data["line_width"][1]) else None
            ),
            line_restfreq=(
                data["line_restfreq"][1][i]
                if i < len(data["line_restfreq"][1])
                else None
            ),
            cont_coeff_1=(
                (data["cont_coeff_1"][1][i])
                if i < len(data["cont_coeff_1"][1])
                else None
            ),
            cont_coeff_2=(
                data["cont_coeff_2"][1][i] if i < len(data["cont_coeff_2"][1]) else None
            ),

            transient_start=(
                data["transient_start"][1][i] if i < len(data["transient_start"][1]) else None
            ),  
            transient_absorb=(
                data["transient_absorb"][1][i] if i < len(data["transient_absorb"][1]) else None
            ),
            transient_ingress=(
                data["transient_ingress"][1][i] if i < len(data["transient_ingress"][1]) else None
            ),
            transient_period=(
                data["transient_period"][1][i] if i < len(data["transient_period"][1]) else None
            ),
        )

        sources.append(source)

    return sources

def load_sources(catfile:File, map_path:File, delimiter:str=" "):
    """_summary_

    Args:
        catfile (File): _description_
        map_path (File): _description_
        delimiter (str, optional): _description_. Defaults to " ".
    """
    catdict = catalogue2dict(catfile, map_path, delimiter)
    
    return dict2sources(catdict) 

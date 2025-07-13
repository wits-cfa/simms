from typing import Dict
from scabha.basetypes import File
import glob
from omegaconf import OmegaConf
import os

# workaround the issue stated in
#  https://github.com/python/mypy/issues/1422
__path__ = os.path.dirname(__file__)


def simms_telescopes() -> Dict:
    """
    Returns a dictionary of known array layouts
    """
    lays = map(File, glob.glob(f"{__path__}/*.geodetic.yaml"))
    laysdict = {}
    for layout in lays:
        # Array name
        arrayinfo = OmegaConf.load(layout)
        allants = list(arrayinfo.antnames)
        all_locations = list(arrayinfo.antlocations)
        anant = len(all_locations)
        
        allsizes = arrayinfo.size
        if isinstance(allsizes, (float, int)):
            allsizes = [allsizes] * anant
        else:
            allsizes = list(allsizes)    
         
        subarrays = arrayinfo.get("subarray", [])
        # add sub-arrays to database
        for subarray in subarrays:
            antnames = arrayinfo.subarray[subarray]
            antlocations = []
            antsizes = []
            for ant in antnames:
                idx = allants.index(ant)
                antlocations.append(all_locations[idx])
                antsizes.append(allsizes[idx])
                
            laysdict[subarray] = dict(
                centre = arrayinfo.centre,
                antlocations = antlocations,
                antnames = antnames,
                size = antsizes,
                coord_sys = arrayinfo.coord_sys,
                mount = arrayinfo.mount,
            )
        
        # add main layout
        if hasattr(arrayinfo, "name"):
            lname = arrayinfo.name 
        else:
            lname = os.path.basename(layout.BASENAME)
            lname = ".".join(lname.split(".")[:-1])
        laysdict[lname] = arrayinfo

    return OmegaConf.create(laysdict)

SIMMS_TELESCOPES = simms_telescopes()
from typing import Dict
from scabha.basetypes import File
import glob
from omegaconf import OmegaConf
import os
import numpy as np

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
                issubarray = True,
            )
        
        # add main layout
        if hasattr(arrayinfo, "name"):
            lname = arrayinfo.name 
        else:
            lname = os.path.basename(layout.BASENAME)
            lname = ".".join(lname.split(".")[:-1])
        laysdict[lname] = arrayinfo

    return OmegaConf.create(laysdict)

def custom_telescopes(layout: str,
                     subarray_list = None,
                     subarray_range = None,
                     subarray_file: File = None) -> Dict:
    """
    Returns a dictionary of a custom array layout.
    """
    lays = glob.glob(f"{__path__}/{layout}.geodetic.yaml")
    laysdict = {}
    
    arrayinfo = OmegaConf.load(lays[0])
    allants = list(arrayinfo.antnames)
    all_locations = list(arrayinfo.antlocations)
    anant = len(all_locations)
        
    allsizes = arrayinfo.size
    if isinstance(allsizes, (float, int)):
        allsizes = [allsizes] * anant
    else:
        allsizes = list(allsizes)    
    
    if subarray_list:
        antnames = subarray_list
        antlocations = []
        antsizes = []
        for ant in antnames:
            idx = allants.index(ant)
            antlocations.append(all_locations[idx])
            antsizes.append(allsizes[idx])

    elif subarray_range:
        
        if len(subarray_range) == 2:
            user_idx = list(range(subarray_range[0], subarray_range[1] + 1))
        elif len(subarray_range) == 3:
            user_idx = list(range(subarray_range[0], subarray_range[1], subarray_range[2]))
        else:
            raise ValueError("Subarray_range must be a list of length 2 or 3.")
        
        antnames = [allants[i] for i in user_idx]
        antlocations = [all_locations[i] for i in user_idx]
        antsizes = [allsizes[i] for i in user_idx]

    elif subarray_file:
        subarray_data = OmegaConf.load(subarray_file)
                
        if 'antnames' in subarray_data:
            antnames = subarray_data['antnames']
            antlocations = []
            antsizes = []
            for ant in antnames:
                idx = allants.index(ant)
                antlocations.append(all_locations[idx])
                antsizes.append(allsizes[idx])

    laysdict = dict(
        centre = arrayinfo.centre,
        antlocations = antlocations,
        antnames = antnames,
        size = antsizes,
        coord_sys = arrayinfo.coord_sys,
        mount = arrayinfo.mount,
        issubarray = True,
        )

    return OmegaConf.create(laysdict)

SIMMS_TELESCOPES = simms_telescopes()

from typing import Dict
import glob
import os

__path__ = [os.path.dirname(__file__)][0]

def get_layout(name):
    """
    Get the specified layout information.
    """
    fname = os.path.join(__path__, f"{name}.geodetic.yaml")
    if os.path.exists(fname):
        return fname
    else:
        raise FileNotFoundError("Layout not part of our known layouts")

def known()-> Dict:
    lays = glob.glob(f"{__path__}/*.geodetic.yaml")
    laysdict = {}
    for layout in lays:
        basename = os.path.basename(layout)
        lname = basename.split(".geodetic.yaml")[0]
        laysdict[lname] = layout
    
    return laysdict


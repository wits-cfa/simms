from typing import Dict
import glob
import os

def get_layout(name):
    """
    Get the array layout.
    """
    fname = os.path.join(__path__, f"{name}.geodetic.yaml")
    if os.path.exists(fname):
        return fname
    raise FileNotFoundError("Layout not part of our known layouts")

def known()-> Dict:
    """
    Returns a dictionary of known array layouts
    """
    lays = glob.glob(f"{__path__}/*.geodetic.yaml")
    laysdict = {}
    for layout in lays:
        basename = os.path.basename(layout)
        lname = basename.split(".geodetic.yaml")[0]
        laysdict[lname] = layout

    return laysdict

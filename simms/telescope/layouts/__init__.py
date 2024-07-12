from typing import Dict
import glob
import os

# workaround the issue stated in
#  https://github.com/python/mypy/issues/1422
__path__ = os.path.dirname(__file__)


def get_layout(name):
    """
    Get the specified layout information.
    """
    fname = os.path.join(__path__, f"{name}.geodetic.yaml")
    if os.path.exists(fname):
        return fname
    raise FileNotFoundError("Layout not part of our known layouts")


def known() -> Dict:
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

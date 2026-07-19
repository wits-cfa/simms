import glob
import os
from typing import Dict

from omegaconf import OmegaConf

thisdir = os.path.dirname(__file__)


def _per_antenna_telescope_names(arrayinfo, nant: int):
    """Per-antenna ``telescope_name`` list, defaulting to the array name.

    Accepts a scalar (broadcast to all antennas) or a per-antenna list, mirroring how
    ``size`` is handled so subarray selection can index it.
    """
    names = arrayinfo.get("telescope_name", "") or arrayinfo.get("name", "")
    if isinstance(names, str):
        return [names] * nant
    return list(names)


def simms_telescopes() -> Dict:
    """
    Returns a dictionary of known array layouts
    """
    lays = map(str, glob.glob(f"{thisdir}/*.geodetic.yaml"))
    laysdict = {}
    for layout in lays:
        # Array name
        arrayinfo = OmegaConf.load(layout)
        allants = list(arrayinfo.antnames)
        all_locations = list(arrayinfo.antlocations)
        anant = len(all_locations)
        ant_to_idx = {name: i for i, name in enumerate(allants)}

        allsizes = arrayinfo.size
        if isinstance(allsizes, (float, int)):
            allsizes = [allsizes] * anant
        else:
            allsizes = list(allsizes)

        alltelnames = _per_antenna_telescope_names(arrayinfo, anant)

        subarrays = arrayinfo.get("subarray", [])
        # add sub-arrays to database
        for subarray in subarrays:
            antnames = arrayinfo.subarray[subarray]
            antlocations = []
            antsizes = []
            anttelnames = []
            for ant in antnames:
                idx = ant_to_idx[ant]
                antlocations.append(all_locations[idx])
                antsizes.append(allsizes[idx])
                anttelnames.append(alltelnames[idx])

            laysdict[subarray] = dict(
                centre=arrayinfo.centre,
                antlocations=antlocations,
                antnames=antnames,
                size=antsizes,
                telescope_name=anttelnames,
                coord_sys=arrayinfo.coord_sys,
                mount=arrayinfo.mount,
                issubarray=True,
            )

        # add main layout
        if hasattr(arrayinfo, "name"):
            lname = arrayinfo.name
        else:
            lname = os.path.basename(layout.BASENAME)
            lname = ".".join(lname.split(".")[:-1])
        laysdict[lname] = arrayinfo

    return OmegaConf.create(laysdict)


def custom_telescopes(layout: str, subarray_list=None, subarray_range=None, subarray_file: str = None) -> Dict:
    """
    Returns a dictionary of a custom array layout.
    """
    layout_file = os.path.join(thisdir, f"{layout}.geodetic.yaml")
    laysdict = {}

    arrayinfo = OmegaConf.load(layout_file)
    allants = list(arrayinfo.antnames)
    all_locations = list(arrayinfo.antlocations)
    anant = len(all_locations)

    allsizes = arrayinfo.size
    if isinstance(allsizes, (float, int)):
        allsizes = [allsizes] * anant
    else:
        allsizes = list(allsizes)

    alltelnames = _per_antenna_telescope_names(arrayinfo, anant)

    if subarray_list:
        ant_to_idx = {name: i for i, name in enumerate(allants)}
        antnames = subarray_list
        antlocations = []
        antsizes = []
        anttelnames = []
        for ant in antnames:
            idx = ant_to_idx[ant]
            antlocations.append(all_locations[idx])
            antsizes.append(allsizes[idx])
            anttelnames.append(alltelnames[idx])

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
        anttelnames = [alltelnames[i] for i in user_idx]

    elif subarray_file:
        subarray_data = OmegaConf.load(subarray_file)

        if "antnames" in subarray_data:
            ant_to_idx = {name: i for i, name in enumerate(allants)}
            antnames = subarray_data["antnames"]
            antlocations = []
            antsizes = []
            anttelnames = []
            for ant in antnames:
                idx = ant_to_idx[ant]
                antlocations.append(all_locations[idx])
                antsizes.append(allsizes[idx])
                anttelnames.append(alltelnames[idx])

    laysdict = dict(
        centre=arrayinfo.centre,
        antlocations=antlocations,
        antnames=antnames,
        size=antsizes,
        telescope_name=anttelnames,
        coord_sys=arrayinfo.coord_sys,
        mount=arrayinfo.mount,
        issubarray=True,
    )

    return OmegaConf.create(laysdict)


SIMMS_TELESCOPES = simms_telescopes()

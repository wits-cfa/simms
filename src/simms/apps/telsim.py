from __future__ import annotations

from types import SimpleNamespace

import shinobi
from dask import config as dask_config
from pydantic import BaseModel, Field

from simms import BIN, set_logger
from simms.telescope import generate_ms, layouts


class SimmsOutputs(BaseModel):
    """Passthrough MS path, so telsim/skysim can be wired into a shinobi Recipe or dosho."""

    ms: str | None = None


def print_data_database(ctx, param, value):
    """
    Display telescope array database
    """
    if value is False:
        return

    for key, val in layouts.SIMMS_TELESCOPES.items():
        info = getattr(val, "info", " --- ")
        if not getattr(val, "issubarray", False):
            print(f"{key}: {info.strip()}")
            subarrays = getattr(val, "subarray", [])
            if subarrays:
                subarray_string = ", ".join(subarrays)
                print(f"  Subarrays: {subarray_string}")
    raise SystemExit()


def _antenna_selection(values, cast=str):
    """Flatten a repeatable antenna-selection option into a plain list, or None if unset.

    shinobi renders the list-typed fields as click ``multiple=True`` options, so the repeated
    form (``-sublist M000 -sublist M005``) arrives as a tuple of separate values, while the
    comma form the help text documents (``-sublist M000,M005``) arrives as one string. Accept
    both. A recipe or cab passing a real YAML list lands here as a list and passes straight
    through.
    """
    if not values:
        return None
    items = []
    for value in values:
        items.extend(part.strip() for part in str(value).split(","))
    return [cast(item) for item in items if item] or None


def runit(opts):
    set_logger(BIN.telsim, opts.log_level)

    # The table writes in generate_ms are dask graphs, so --nworkers has to reach the
    # scheduler the same way skysim sets it; without this the option was accepted,
    # documented and silently ignored.
    dask_config.set(scheduler="threads", num_workers=opts.nworkers)

    msname = opts.ms
    telescope = opts.telescope
    direction = opts.direction.split(",")
    starttime = opts.starttime
    dtime = opts.dtime
    ntimes = opts.ntime
    startfreq = opts.startfreq
    dfreq = opts.dfreq
    nchan = opts.nchan
    correlations = opts.correlations.split(",")
    rowchunks = opts.rowchunks
    sefd = opts.sefd
    tsys_over_eta = opts.tsys_over_eta
    column = opts.column
    startha = opts.startha
    l_src_limit = opts.low_source_limit
    h_src_limit = opts.high_source_limit
    freq_range = opts.freq_range
    sfile = opts.sensitivity_file
    if freq_range is not None:
        freq_range = freq_range.split(",")
    subarray_list = _antenna_selection(opts.subarray_list)
    subarray_range = _antenna_selection(opts.subarray_range, cast=int)
    subarray_file = opts.subarray_file
    smooth = opts.smooth
    fit_order = opts.fit_order

    generate_ms.create_ms(
        ms=msname,
        telescope_name=telescope,
        pointing_direction=direction,
        dtime=dtime,
        ntimes=ntimes,
        start_freq=startfreq,
        dfreq=dfreq,
        nchan=nchan,
        correlations=correlations,
        row_chunks=rowchunks,
        sefd=sefd,
        column=column,
        smooth=smooth,
        fit_order=fit_order,
        start_time=starttime,
        start_ha=startha,
        freq_range=freq_range,
        sfile=sfile,
        tsys_over_eta=tsys_over_eta,
        subarray_list=subarray_list,
        subarray_range=subarray_range,
        subarray_file=subarray_file,
        low_source_limit=l_src_limit,
        high_source_limit=h_src_limit,
        telescope_name_column=opts.telescope_name_column,
    )


@shinobi.pystep(name=BIN.telsim, info="Create an empty Measurement Set from a telescope layout.")
def telsim(
    ms: str = Field(..., description="Observation name/id/label"),
    telescope: str = Field(
        ..., description="Name of telescope you are simulating", json_schema_extra={"abbreviation": "tel"}
    ),
    subarray_list: list[str] | None = Field(
        None,
        description="Custom list of antennas to use, e.g., M000,M005,SKA009. "
        "Must be a subarray of the given telescope.",
        json_schema_extra={"abbreviation": "sublist"},
    ),
    # `str` deliberately leads the union: shinobi picks the click type from the first
    # int/float/bool/str leaf, so a bare `list[int]` renders as `INTEGER` and click rejects
    # the documented comma form ("'0,64' is not a valid integer") before the value ever
    # reaches us. Leading with `str` gives a STRING option that takes both `0,64` and
    # `-subrange 0 -subrange 64`, while the `int` arm still accepts a YAML list of ints from
    # a recipe. `_antenna_selection` casts to int. Narrowing this to `list[int]` re-breaks
    # the CLI -- see test_subarray_range_accepts_comma_separated_indices.
    subarray_range: list[str | int] | None = Field(
        None,
        description="Custom range of antenna indices to use, e.g. start,end,step (step optional; "
        "end is inclusive when no step is given). Must be a subarray of the given telescope.",
        json_schema_extra={"abbreviation": "subrange"},
    ),
    subarray_file: str | None = Field(
        None,
        description="File listing custom antennas to use (antnames key, e.g. [M000,M005,SKA009]). "
        "Must be a subarray of the given telescope.",
        json_schema_extra={"abbreviation": "subfile"},
    ),
    telescope_name_column: str = Field(
        "TELESCOPE_NAME",
        description="Name of the ANTENNA-table column that holds the per-antenna telescope/type label "
        "(used by skysim to select a primary beam).",
        json_schema_extra={"abbreviation": "tnc"},
    ),
    direction: str = Field(
        "J2000,1h0m0s,-31d0m0s",
        description="Direction of field centre for MS, e.g. J2000,0h24m20s,-30d12m33s.",
        json_schema_extra={"abbreviation": "dir"},
    ),
    starttime: str | None = Field(
        None,
        description="Observation start time in UTC, e.g. '2024-03-14T06:15:10'. Default is the current machine time.",
        json_schema_extra={"abbreviation": "st"},
    ),
    startha: float | None = Field(
        None,
        description="Hour angle at start of observation. Can be used instead of date.",
        json_schema_extra={"abbreviation": "sha"},
    ),
    dtime: float = Field(
        8, description="Integration/exposure time in seconds.", json_schema_extra={"abbreviation": "dt"}
    ),
    ntime: int = Field(10, description="Number of time slots for MS.", json_schema_extra={"abbreviation": "nt"}),
    startfreq: str | float = Field(
        "1420MHz",
        description="Centre of first frequency channel, e.g 0.55GHz. Hertz assumed if no units.",
        json_schema_extra={"abbreviation": "sf"},
    ),
    dfreq: str | float = Field(
        "1MHz",
        description="Channel width, e.g 2.4MHz. Hertz assumed if no units.",
        json_schema_extra={"abbreviation": "df"},
    ),
    nchan: int = Field(9, description="Number of frequency channels.", json_schema_extra={"abbreviation": "nc"}),
    correlations: str = Field(
        "XX,YY", description="Feed correlations for MS, e.g., 'XX,YY'.", json_schema_extra={"abbreviation": "corr"}
    ),
    nworkers: int = Field(4, description="Number of workers (one per CPU)."),
    rowchunks: int = Field(
        50000,
        description="Number of chunks to divide the data into; more chunks improves computation speed.",
        json_schema_extra={"abbreviation": "rc"},
    ),
    column: str = Field(
        "MODEL_DATA",
        description="The column in which to corrupt the visibilities with noise.",
        json_schema_extra={"abbreviation": "col"},
    ),
    sefd: float | None = Field(None, description="Antenna SEFD (one value for all frequencies)."),
    tsys_over_eta: float | None = Field(
        None,
        description="Antenna system temperature over aperture efficiency (one value for all frequencies).",
        json_schema_extra={"abbreviation": "tos"},
    ),
    sensitivity_file: str | None = Field(
        None,
        description="File with antenna spectral sensitivity info. Keys: 'freq, tsys, sefd, tsys_over_eta'.",
        json_schema_extra={"abbreviation": "sfile"},
    ),
    low_source_limit: float | None = Field(
        None,
        description="Minimum reliable source elevation (deg); data below this is flagged.",
        json_schema_extra={"abbreviation": "lsl"},
    ),
    high_source_limit: float | None = Field(
        None,
        description="Maximum reliable source elevation (deg); data above this is flagged.",
        json_schema_extra={"abbreviation": "hsl"},
    ),
    freq_range: str | None = Field(
        None,
        description="A list of start frequency, end frequency, and number of channels, e.g. startfreq,endfreq,nchan.",
        json_schema_extra={"abbreviation": "fr"},
    ),
    smooth: str | None = Field(
        None,
        description="SEFD fitting option when a sensitivity file is given: 'polyn' or 'spline'.",
    ),
    fit_order: int | None = Field(
        None,
        description="Fitting order used when approximating the MS-frequency SEFDs.",
        json_schema_extra={"abbreviation": "fo"},
    ),
    log_level: str = Field("INFO", description="Logging verbosity."),
) -> SimmsOutputs:
    opts = SimpleNamespace(**locals())
    runit(opts)
    return SimmsOutputs(ms=ms)

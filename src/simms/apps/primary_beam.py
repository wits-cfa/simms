from __future__ import annotations

from types import SimpleNamespace
from typing import Literal

import shinobi
from pydantic import BaseModel, Field

from simms import BIN, set_logger


class PrimaryBeamOutputs(BaseModel):
    """Passthrough output path of the primary-beam operation (beam FITS or beamed sky model)."""

    output: str | None = None


def _require(opts, field):
    if not getattr(opts, field, None):
        raise RuntimeError(f"--{field.replace('_', '-')} is required for mode {opts.mode!r}.")


def runit(opts):
    set_logger(BIN.primary_beam, opts.log_level)
    from simms.skymodel import pb_ops

    mode = opts.mode
    if mode == "to-fits":
        _require(opts, "beam_pattern")
        pb_ops.to_fits(opts)
    elif mode == "tag-ms":
        _require(opts, "ms")
        pb_ops.tag_ms(opts)
    elif mode in ("apply", "correct"):
        _require(opts, "ms")
        _require(opts, "beam_pattern")
        if sum(bool(x) for x in (opts.fits_sky, opts.ascii_sky)) != 1:
            raise RuntimeError("apply/correct needs exactly one of --fits-sky or --ascii-sky.")
        invert = mode == "correct"
        if opts.fits_sky:
            pb_ops.apply_correct_image(opts, invert)
        else:
            pb_ops.apply_correct_ascii(opts, invert)
    else:
        raise RuntimeError(f"Unknown primary-beam mode {mode!r}.")


@shinobi.pystep(
    name=BIN.primary_beam, info="Primary-beam utilities (build/tag/apply/correct); no visibility simulation."
)
def primary_beam(
    mode: Literal["to-fits", "tag-ms", "apply", "correct"] = Field(..., description="Operation to perform."),
    beam_pattern: str | None = Field(
        None,
        description="Beam model: a cosine-taper CSV path, a built-in name, a band shorthand (L/UHF), or a FITS cube.",
        json_schema_extra={"abbreviation": "bp"},
    ),
    beam_band: Literal["UHF", "L"] = Field(
        "L", description="Default band for a built-in beam when beam-pattern omits one."
    ),
    beam_pa_step: float = Field(
        1.0, description="Parallactic-angle sampling step (degrees) for the time-averaged beam."
    ),
    ms: str | None = Field(
        None,
        description="Measurement set (time/PA range, array position and frequencies). "
        "Required for tag-ms/apply/correct.",
    ),
    fits_sky: str | None = Field(
        None, description="Input FITS image sky model (apply/correct).", json_schema_extra={"abbreviation": "fits"}
    ),
    ascii_sky: str | None = Field(
        None,
        description="Input ASCII component sky model (apply/correct).",
        json_schema_extra={"abbreviation": "ascii"},
    ),
    ascii_delimiter: str | None = Field(
        None,
        description="Delimiter used in the ascii-sky file. Defaults to whitespace.",
        json_schema_extra={"abbreviation": "ad"},
    ),
    source_schema: str | None = Field(
        None,
        description="Custom source schema (YAML) mapping the ascii-sky columns to the fields simms expects.",
    ),
    output: str | None = Field(
        None,
        description="Output path - FITS beam (to-fits) or beamed/corrected sky model (apply/correct).",
        json_schema_extra={"abbreviation": "o"},
    ),
    telescope_name_column: str = Field(
        "TELESCOPE_NAME",
        description="ANTENNA-table column holding the per-antenna telescope/type label (tag-ms).",
        json_schema_extra={"abbreviation": "tnc"},
    ),
    label: str | None = Field(None, description="Single telescope-name label applied to all antennas (tag-ms)."),
    label_map: str | None = Field(None, description="YAML mapping antenna NAME -> telescope-name label (tag-ms)."),
    from_layout: str | None = Field(
        None,
        description="simms layout whose per-antenna telescope_name is matched to the MS antenna names (tag-ms).",
    ),
    pb_cutoff: float = Field(0.1, description="In correct mode, blank (NaN) where the beam is below this level."),
    field_id: int = Field(
        0,
        description="FIELD_ID whose phase centre and time span define the beam (apply/correct).",
        json_schema_extra={"abbreviation": "fi"},
    ),
    spw_id: int = Field(
        0, description="Spectral-window (DATA_DESC_ID) whose frequencies define the beam (apply/correct)."
    ),
    pixel_size: str = Field("1arcmin", description="Angular pixel size for the to-fits grid, e.g. '1arcmin'."),
    npix: int = Field(256, description="Number of pixels per side for the to-fits grid."),
    start_freq: str | None = Field(
        None,
        description="Start frequency of the to-fits cube, e.g. '856MHz'. Defaults to the beam's first frequency.",
        json_schema_extra={"abbreviation": "sf"},
    ),
    chan_width: str | None = Field(
        None,
        description="Channel width of the to-fits cube, e.g. '10MHz'. "
        "Defaults to spanning the beam range across nchan.",
        json_schema_extra={"abbreviation": "cw"},
    ),
    nchan: int | None = Field(
        None,
        description="Number of output channels in the to-fits cube. Defaults to the beam's channel count.",
        json_schema_extra={"abbreviation": "nc"},
    ),
    nworkers: int = Field(4, description="Number of worker threads."),
    log_level: str = Field("INFO", description="Logging verbosity."),
) -> PrimaryBeamOutputs:
    opts = SimpleNamespace(**locals())
    runit(opts)
    return PrimaryBeamOutputs(output=output)

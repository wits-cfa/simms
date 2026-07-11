from simms import BIN, set_logger


def _require(opts, field):
    if not getattr(opts, field, None):
        raise RuntimeError(f"--{field.replace('_', '-')} is required for mode {opts.mode!r}.")


def runit(opts):
    set_logger(BIN.primary_beam, opts["log_level"])
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

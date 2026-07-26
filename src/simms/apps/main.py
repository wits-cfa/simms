from __future__ import annotations

import click
from shinobi.clickutil import build_options, unflatten_kwargs
from shinobi.steps.dispatch import _dispatch

from simms import BIN, __version__
from simms.apps import primary_beam, skysim, telsim


class RemoveMSIfChained(click.Command):
    """
    A custom command class that removes the 'ms' parameter based on a parent flag.
    """

    def make_parser(self, ctx):
        """
        This method is called before parsing. It checks the parent context
        for a '--chain' flag and removes the 'ms' argument if it's found.
        """
        if ctx.parent and ctx.parent.params.get("chain", False):
            self.params = [p for p in self.params if p.name != "ms"]
        return super().make_parser(ctx)


# Long-flag aliases, keyed by subcommand and field name.
#
# telsim predates skysim and primary-beam and spells three shared concepts differently:
# --rowchunks vs --row-chunks, --startfreq vs --start-freq, --dfreq vs --chan-width. Renaming
# either side would break existing scripts and Stimela recipes, and telsim's spellings are
# internally symmetric (--startfreq/--starttime, --dfreq/--dtime), so neither side is simply
# wrong. Each spelling is instead accepted on both commands and resolves to the same field, so
# a flag that works on one subcommand is never rejected by another. `--help` lists both.
FLAG_ALIASES = {
    "telsim": {
        "rowchunks": "--row-chunks",
        "startfreq": "--start-freq",
        "dfreq": "--chan-width",
    },
    "skysim": {
        "row_chunks": "--rowchunks",
    },
    "primary-beam": {
        "start_freq": "--startfreq",
        "chan_width": "--dfreq",
    },
}


def _add_alias(opt, alias):
    """Add a second long flag to an already-built click Option.

    click's parser reads `Parameter.opts` when the command is invoked, while the callback
    kwarg name was derived from the primary flag at construction, so appending here adds a
    spelling without touching the round-trip back to the model field.
    """
    if alias not in opt.opts:
        opt.opts.append(alias)


def _make_command(step, *, positional, chained, extra_options=()):
    """Build a `click.Command` for a `@shinobi.pystep` StepRef.

    Options come from the step's pydantic ``inputs_model`` via shinobi's
    ``build_options`` (choices, abbreviations, bool/list handling). The
    ``log_level`` field is dropped -- the root group's ``--log-level``
    controls it -- and the ``positional`` field is rendered as a
    ``click.Argument`` (build_options only emits ``--options``), and any
    field in ``FLAG_ALIASES`` gains a second long flag. The
    callback re-nests the flat kwargs and dispatches the step in-process
    via shinobi, exactly as ``shinobi.cli``'s ``run`` command does.
    """
    options = [opt for opt in build_options(step.step.inputs_model) if opt.name != "log_level"]

    aliases = FLAG_ALIASES.get(step.step.name, {})
    for opt in options:
        if opt.name in aliases:
            _add_alias(opt, aliases[opt.name])

    params = list(extra_options)
    for opt in options:
        if opt.name == positional:
            params.append(click.Argument([positional], required=True, type=opt.type))
        else:
            params.append(opt)

    def _callback(**raw):
        ctx = click.get_current_context()
        kwargs = unflatten_kwargs(step.step.inputs_model, raw)
        kwargs["log_level"] = ctx.obj["log_level"]
        if chained and ctx.obj["chain"]:
            kwargs["ms"] = ctx.obj["ms"]
        result = _dispatch(step.step, step.func, **kwargs)
        if not result.success:
            raise click.ClickException(f"{step.step.name!r} failed (returncode {result.returncode}).")

    cls = RemoveMSIfChained if chained else click.Command
    return cls(
        name=step.step.name,
        params=params,
        callback=_callback,
        help=step.step.info,
        no_args_is_help=True,
    )


@click.group(chain=True, no_args_is_help=True)
@click.version_option(str(__version__))
@click.option("--ms", "-ms")
@click.option(
    "--log-level", "-ll", help="Log level", type=click.Choice(["INFO", "WARNING", "CRITICAL", "ERROR"]), default="INFO"
)
@click.option(
    "--chain",
    is_flag=True,
    help="Chain telsim and skysim for an End-to-End simulation."
    " When this option is set, the -ms/--ms has to be given in the "
    "main command and excluded from both sub-commands",
)
@click.pass_context
def cli(ctx, ms, log_level, chain):
    """
    Tools for simulating radio interferometry observations. 'telsim' creates a simulated observation
    (Measurement Set; MS), and 'skysim' populates an MS with visibilities generated from a given skymodel
    (FITS or ASCII format). For more info on the tools run:

        simms telsim --help

        simms skysim --help

    """

    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level
    ctx.obj["chain"] = chain
    if chain:
        if ms:
            ctx.obj["ms"] = ms
        else:
            raise click.exceptions.MissingParameter(
                "The --ms/-ms option is required when --chain is set", param_type="Option", param_hint="--ms"
            )


_list_option = click.Option(
    ["--list", "-ls"],
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=telsim.print_data_database,
    help="Displays a list of available telescope array layouts",
)

cli.add_command(
    _make_command(telsim.telsim, positional="ms", chained=True, extra_options=[_list_option]), name=BIN.telsim
)
cli.add_command(_make_command(skysim.skysim, positional="ms", chained=True), name=BIN.skysim)
cli.add_command(_make_command(primary_beam.primary_beam, positional="mode", chained=False), name=BIN.primary_beam)

# simms

Radio-interferometry simulator. `telsim` builds a Measurement Set from a telescope layout;
`skysim` predicts model visibilities from a sky model into an MS; `primary-beam` provides
standalone beam operations. Single CLI entry point `simms` (`simms.apps.main:cli`), with
subcommands wired in `src/simms/apps/` (one module + one `<name>.yaml` cab per subcommand).
Src layout: the importable package lives under `src/simms/`, not at the repo root.

## Environment & commands

Use `uv` for everything — never call `python`/`pytest`/`ruff` directly.

- Run code: `uv run python ...`, or the CLI via `uv run simms <subcommand> ...`
- Tests: `uv run --group tests python -m pytest` (a specific file: `... python -m pytest tests/<name>_tests.py`)
- Lint/format: `uv run --group ruff ruff check src tests` and `uv run --group ruff ruff format <paths>`

A pre-commit hook runs `ruff` + `ruff-format`; if it reformats a file the commit aborts, so
re-stage and commit again.

## Tests

- Test files must be named `*_tests.py` (pytest is configured with `python_files = ["*_tests.py"]`);
  a `foo_test.py` or `test_foo.py` will not be collected.
- Temp MSs/files/dirs go through `tests.InitTest` (`random_named_file` / `random_named_directory`),
  which registers them for cleanup — don't hand-roll `tempfile`.
- Heavy or optional dependencies are opt-in dependency groups and guarded with
  `pytest.importorskip`, so the default `tests` run stays light. Example: the CASA round-trip
  test needs the `casa` group — `uv run --group tests --group casa python -m pytest tests/casa_roundtrip_tests.py`.

## MS conventions (load-bearing, easy to get wrong)

- **Metadata has a single authoritative source; never infer it.** The per-antenna telescope/type
  label lives in the `ANTENNA` table column named by `--telescope-name-column` (default
  `TELESCOPE_NAME`). Read it and fail clearly if absent — do not guess from `DISH_DIAMETER` etc.
- **Pointing vs phase centre are different.** `FIELD.PHASE_DIR` is the correlator phase-tracking
  centre (arbitrary, shiftable). The primary beam is centred on the antenna pointing centre in
  `POINTING.DIRECTION`. Use `simms.skymodel.beams.read_pointing_centre` for the beam centre.
- **`SPECTRAL_WINDOW.MEAS_FREQ_REF` must be set** (5 == TOPO). casacore defaults it to 0 (REST),
  which leaves the spectral frame undefined and makes CASA imaging fail ("No MeasFrame specified
  for conversion of Frequency").
- **casacore STRING columns are numpy `object` dtype**, written in one chunk
  (`da.from_array(values, chunks=n)`). Adding a *new* column to a standard subtable needs an
  explicit descriptor, e.g. `xds_to_table(..., "{ms}::ANTENNA", columns=[col], descriptor="mssubtable('ANTENNA')")`.

## Beam data

Cosine-taper (`beams.py`) tables under `src/simms/skymodel/beam_data/`. The `MKAT-AA-*` model and its
tables are vendored from katbeam (BSD-3-Clause) — keep that attribution in `beam_data/NOTICE`. The
other tables ship as ordinary bundled package data.

## Git

- Branch off `main` for changes; open PRs against `main` (repo `wits-cfa/simms`).
- End commit messages with: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`
- `gh pr edit --body` can fail on this repo with a Projects-classic GraphQL error; edit the body
  via `gh api -X PATCH repos/wits-cfa/simms/pulls/<n> -f body=@file` instead.

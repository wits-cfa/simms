# Contributing to simms

Thanks for your interest in contributing! **simms** is a radio-interferometry
simulator: `telsim` builds a Measurement Set from a telescope layout, `skysim`
predicts model visibilities from a sky model into it, and `primary-beam`
provides standalone beam operations.

By participating you agree to abide by our
[Code of Conduct](https://github.com/wits-cfa/simms/blob/main/CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report bugs** and request features via [issues](https://github.com/wits-cfa/simms/issues).
- **Improve documentation** under `docs/` or the docstrings that feed the API
  reference.
- **Submit code** -- bug fixes, new telescope layouts, sky-model formats,
  tests.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) for everything -- avoid
calling `python`/`pytest`/`ruff` directly:

```bash
git clone https://github.com/wits-cfa/simms.git
cd simms
uv run --group tests python -m pytest
uv run --group ruff ruff check src tests
```

## Testing

Run the default suite with:

```bash
uv run --group tests python -m pytest
```

- Test files must be named `*_tests.py` (pytest is configured with
  `python_files = ["*_tests.py"]`) -- a `test_foo.py` will not be collected.
- Temp MSs/files/dirs should go through `tests.InitTest`
  (`random_named_file`/`random_named_directory`), which registers them for
  cleanup, rather than hand-rolled `tempfile` usage.
- Heavy or optional dependencies are opt-in dependency groups, guarded with
  `pytest.importorskip`, so the default run stays light. For example, the
  CASA round-trip test needs the `casa` group:

  ```bash
  uv run --group tests --group casa python -m pytest tests/casa_roundtrip_tests.py
  ```

**New features and bug fixes should come with tests.**

## Code style

- **Lint must be clean**: `ruff check src tests` should report no errors.
- `ruff format` is available for autoformatting; a pre-commit hook runs both
  `ruff` and `ruff-format` -- if it reformats a file the commit aborts, so
  re-stage and commit again.
- Use type hints and docstrings on public API -- they render into the Sphinx
  API reference via autodoc.
- Match the surrounding code's naming, comment density, and idiom.

### MS conventions

The package works with `casacore` Measurement Sets and has a few load-bearing
conventions around metadata (telescope-name column, phase vs. pointing
centre, spectral frame) -- see the *MS conventions* section of
[`CLAUDE.md`](https://github.com/wits-cfa/simms/blob/main/CLAUDE.md) (also
readable as `AGENTS.md`) and the
[MS conventions doc page](https://simms.readthedocs.io/en/latest/concepts/ms-conventions.html)
before touching MS I/O code.

## Documentation

Docs are built with Sphinx (Furo theme) and hosted on Read the Docs. Build
them locally with:

```bash
uv sync --group docs
uv run sphinx-build -b html docs docs/_build/html
```

Please update the docs when you change public API. If you add a
documentation dependency, keep `docs/requirements.txt` in sync with the
`docs` dependency group in `pyproject.toml` (Read the Docs installs from the
former).

## Pull requests

1. Branch off `main` and keep PRs **small and focused**.
2. Make sure `pytest` and `ruff check src tests` pass locally, and that docs
   build if you touched public API.
3. Push and open a PR against `main`. Reference any related issue
   (e.g. "Closes #12").
4. CI must be green.

### Commit messages

Write clear, descriptive commit messages explaining *why* a change is made.

## Versioning and releases

The project follows [Semantic Versioning](https://semver.org/). Releases are
maintainer-driven: the maintainer bumps `version` in `pyproject.toml` and
publishes a GitHub release, which triggers the publish workflow.

## License

By contributing, you agree that your contributions are licensed under the
project's [GPL-3.0 License](https://github.com/wits-cfa/simms/blob/main/LICENSE).

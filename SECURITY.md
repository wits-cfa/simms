# Security Policy

## Supported versions

simms is pre-1.0 software. Security fixes are applied to the latest `3.0betaN`
release only; there are no long-term-support branches yet.

| Version        | Supported          |
| -------------- | ------------------- |
| latest `3.0beta*` | :white_check_mark: |
| older          | :x:                 |

## Reporting a vulnerability

**Please do not report security issues in public GitHub issues.**

Report vulnerabilities privately by email to **sphemakh@gmail.com** (or via
GitHub's [private vulnerability reporting][ghsa] on this repository, if
enabled). Include enough detail to reproduce -- affected version, the
command run (`telsim`/`skysim`/`primary-beam`), inputs, and the impact you
observed.

We aim to acknowledge reports within a reasonable time, work with you on a
fix, and credit you in the release notes if you'd like.

[ghsa]: https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability

## Security posture

simms reads user-supplied sky models (ASCII catalogues, FITS images, WSClean
component lists) and telescope layout/schema YAML, and reads/writes
`casacore` Measurement Sets. In short:

- Sky model and layout files are parsed as data (`astropy`/`numpy`/YAML
  loaders), never `eval`/`exec`-ed as code.
- simms does not shell out to external commands or invoke `subprocess`/
  `os.system` -- there is no cab/orchestration layer here (that's a
  different project, [stimela](https://github.com/caracal-pipeline/stimela)).
- MS I/O goes through `dask-ms`/`python-casacore`; simms treats
  MS/FITS/catalogue metadata as untrusted input and expects it to fail
  clearly (not silently misinterpret) when required fields are missing -- see
  the *MS conventions* section of [`CLAUDE.md`](CLAUDE.md).

If you find a way around any of these guarantees, it's a security issue --
please report it as above.

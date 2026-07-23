### Unreleased

- Fast image-domain a-term (primary-beam) correction for the FITS-image
  prediction path (`simms.skymodel.aterms`), in the spirit of WSClean's IDG and
  DDFacet's facet beams but with no spatial approximation: per-antenna beams are
  applied to the full image per baseline-type class, interpolated linearly in
  time on the parallactic-angle grid (algebraically identical to the exact
  per-component DFT kernels, asserted in the tests) and between adaptively
  chosen frequency knots (`--aterm-freq-tol`; `0` samples every channel). This
  replaces the PA-averaged single-antenna power beam as the default
  (`--fits-beam-mode aterm`); the legacy approximation remains available as
  `--fits-beam-mode average`, and is the automatic fallback for diagonal beams
  on a circular-correlation MS. `--beam-jones full` is now honoured on the
  FITS-image path. When the DFT backend wins the FITS cost model under a beam,
  the model is bridged to the exact per-component beam kernels instead of the
  approximate image beam.
- A-term review follow-ups: an image too large for the voltage-map cache now
  degrades to `--fits-beam-mode average` with a warning instead of raising
  `MemoryError` (the ceiling is still enforced for direct API callers, and its
  message names that escape); apparent-beam products are yielded one correlation
  at a time rather than materialised per pass, and memoised across a channel
  segment in diagonal mode; `attach_fits_aterm` and `component_sky_from_fits_dft`
  reject a circular-basis model rather than silently producing wrong cross-hands;
  the planned gridder-pass count is logged at DEBUG.

### 3.0.0 -> 3.0.1

- gitingore poetry|uv.lock
- Port cabs to shinobi/dosho
- Fix `-pb` duplicate abbrevation. The `primary-beam` option retains it
  and `predict-backend` looses it
- `primary-beam` declares `ms` alongside `output` in its pystep outputs, so
  `tag-ms` (which mutates the MS in place) can be chained from in a recipe


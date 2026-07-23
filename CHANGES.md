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

### 3.0.0 -> 3.0.1

- gitingore poetry|uv.lock
- Port cabs to shinobi/dosho
- Fix `-pb` duplicate abbrevation. The `primary-beam` option retains it
  and `predict-backend` looses it
- `primary-beam` declares `ms` alongside `output` in its pystep outputs, so
  `tag-ms` (which mutates the MS in place) can be chained from in a recipe


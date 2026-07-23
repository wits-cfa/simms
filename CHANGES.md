### 3.0.1 -> unreleased

- `skysim`: `--row-chunks` is now an upper bound rather than a fixed chunk size.
  A fixed size tied the task count to the length of the MS, so a short track
  produced fewer chunks than workers and left most of them idle (76608 rows at
  the 10000-row default is 8 chunks, pinning `--nworkers 32` to ~8 cores). The
  size is now reduced so each worker gets several chunks, floored at 256 rows.
  Measured on MeerKAT: 2.4x faster on a 5-min/10k-source predict (67.0s -> 28.0s,
  720% -> 2420% CPU), 1.1-1.4x on longer tracks. Chunk size is never increased,
  so memory per task cannot grow. Note that, as before, the thermal-noise
  realisation for a given `--seed` depends on the chunking, so a `--sefd` run
  reproduces a previous realisation only if `--row-chunks` is set explicitly.

### 3.0.0 -> 3.0.1

- gitingore poetry|uv.lock
- Port cabs to shinobi/dosho
- Fix `-pb` duplicate abbrevation. The `primary-beam` option retains it
  and `predict-backend` looses it
- `primary-beam` declares `ms` alongside `output` in its pystep outputs, so
  `tag-ms` (which mutates the MS in place) can be chained from in a recipe


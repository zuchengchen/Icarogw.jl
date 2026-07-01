# Integration Tests

`test/integration/runtests.jl` is reserved for public, unauthenticated
real-data workflows that are too slow, network-dependent, or data-heavy for the
default offline `Pkg.test()` path.

Large downloaded data must not be committed. Use a cache directory outside
tracked fixtures. By default the integration runner uses:

```text
test/integration/data/
```

Set `ICAROGW_INTEGRATION_CACHE` to override the cache location. Set
`ICAROGW_RUN_PUBLIC_INTEGRATION=1` to opt in to network/data tests once suites
are registered.

The current runner is a scaffold. Concrete suites should be added as catalog,
public skymap, stochastic/OmegaGW, and joint-likelihood workflows become ready
for public-data coverage.

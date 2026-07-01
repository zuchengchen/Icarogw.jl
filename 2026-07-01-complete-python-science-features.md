# Goal: Complete Python Science Features In Icarogw.jl

## Goal Mode Objective

Follow the saved goal file at `/home/czc/projects/working/Icarogw.jl/2026-07-01-complete-python-science-features.md`; complete the task only when the verification section passes, and stop to ask if any listed stop condition occurs.

## Full Prompt

### Objective

Make `/home/czc/projects/working/Icarogw.jl` a Julia-native scientific replacement for the local Python reference package at `../icarogw`, covering all Python scientific functionality while preserving Julia-native API design, numerical validation, documentation, examples, and review evidence.

### Context

The current Julia package already implements the first core vertical slice: cosmology, conversions, priors, rate models, data containers, simulation, likelihoods, plotting, and Dynesty-compatible closures. The Python reference still contains major scientific features not yet implemented in Julia, especially catalog/skymap/dark-siren/bright-siren/EM workflows, stochastic background, `Omega_GW`, joint CBC+stochastic likelihoods, additional utilities, file-format compatibility, and full tutorial-level documentation.

Use local `../icarogw` as the source of truth. Do not chase upstream changes unless needed for clarification. Formal package code must remain native Julia and must not depend on Python bridges.

### Brainstorming Direction

Use a staged module-by-module migration. First audit Python-to-Julia gaps and build reference fixture generation. Then migrate and verify modules in phases: conversions/skymap helpers, catalog and EM workflows, stochastic/OmegaGW, rates/likelihood integration, migration tools, observability/cache/workspace abstractions, docs/examples/tutorials, release notes. Commit after stable, tested module phases; never push automatically.

### Discovery Summary

Reference version is local `../icarogw`. API may be breaking and should be pure Julia-native, with Python-to-Julia mapping documented instead of mirroring Python names. Numerical alignment is a primary success criterion: generate Python reference fixtures where practical, compare normal operating regimes, and document intentional differences where Python behavior is unstable or scientifically suspect. Default tests should be offline; public-data integration tests may download/cache data separately. Julia compatibility must remain `1.10`.

### Scope

Implement Python scientific functionality including catalog preprocessing, pixelated catalogs, redshift grids, interpolation, `icarogw_catalog`/`galaxy_catalog`-equivalent workflows, skymap/FITS/HEALPix-style helpers, k-corrections, EM likelihood helpers, dark-siren and bright-siren workflows, stochastic background, `Omega_GW`, `dEdf`, omega weights, spectral sirens, stochastic-only likelihoods, CBC+stochastic joint likelihoods, non-Condor utilities, general bounds helpers, migration/reading tools for Python workflow HDF5/FITS/CSV/NPZ-like outputs, workspace/cache/precompute abstractions, logging/diagnostics/progress/run reports, examples, tutorials, API mapping, migration notes, and release-note/changelog drafts.

Allow mature Julia-native dependencies compatible with Julia 1.10. External command-line tools may be used for development or fixture generation when Julia ecosystem gaps exist, but formal package code must not use Python bridges. Create or reorganize directories such as `src/Catalog/`, `test/integration/`, `scripts/fixtures/`, and `docs/tutorials/` when useful.

### Out Of Scope

Do not implement Condor/HTCondor helpers, pickle support, Python bridge dependencies in formal package code, CuPy/GPU backend switching, automatic push, Julia registry publication, private or authenticated data downloads, a CLI system, or a configuration-driven workflow system.

### Verification

Run and pass `julia --project=. -e 'using Pkg; Pkg.test()'`. Add or update offline unit tests and small committed fixtures for core formulas, containers, file readers, catalog/skymap pieces, stochastic/OmegaGW, likelihoods, diagnostics, cache behavior, and migration tools.

Provide reproducible Python fixture generation scripts and environment notes, without making Python a Julia package dependency. Add or update `test/integration/runtests.jl` for public, unauthenticated real-data workflows and cache downloaded data outside committed large files. Run `julia --project=. test/integration/runtests.jl` when network/data availability permits, and document skips clearly.

Run `julia --project=. benchmark/benchmarks.jl` and update benchmark coverage for core likelihood, catalog, stochastic weights, and representative workflows. Do not enforce hard runtime thresholds unless the user later asks.

Run representative examples/tutorial scripts, including dark siren, bright siren/EM, stochastic/OmegaGW, joint likelihood, and Dynesty-compatible workflows. Update README, `docs/api_mapping.md`, `docs/migration_notes.md`, tutorial docs, notebook-oriented README, examples README, and release-note/changelog draft.

Before marking complete, produce a module-by-module review checklist covering Python source coverage, Julia API mapping, numerical fixture evidence, integration evidence, known differences, risks, and remaining limitations.

### Stop Conditions

Stop and ask the user before making major scientific/API/dependency trade-offs, using a nontrivial external tool as an ongoing workflow requirement, accepting unclear Python behavior as normative, depending on data with unclear license or authentication requirements, dropping Python file-format compatibility for an important workflow, changing Julia minimum version, introducing a formal Python bridge, or declaring completion with failed or unrun required verification.

## Notes

- Created for Codex Goal mode.
- Do not mark complete until the verification section passes or the user explicitly changes the completion standard.
- Stage commits by module phase after relevant tests pass; do not push.

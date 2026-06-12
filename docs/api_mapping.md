# Python-To-Julia API Mapping

This document tracks first-version migration status from `../icarogw/icarogw`
to native Julia `Icarogw.jl`.

Status meanings:

- `implemented`: native Julia implementation exists in first-version scope.
- `renamed`: functionality exists under Julia-style names.
- `merged`: Python concepts are combined into a Julia-native abstraction.
- `planned`: intentionally deferred with clear placeholder behavior.
- `excluded`: intentionally not migrated.

## cosmology.py

| Python API | Julia API | Status | Notes |
| --- | --- | --- | --- |
| `base_cosmology.z2dl` | `luminosity_distance` | implemented/renamed | Mpc distances. |
| `base_cosmology.dl2z` | `redshift_at_luminosity_distance` | implemented/renamed | Root-based inversion. |
| `base_cosmology.z2Vc` | `comoving_volume` | implemented/renamed | Full-sky Gpc^3. |
| `base_cosmology.dVc_by_dzdOmega_at_z` | `dvc_dz_dOmega` | implemented/renamed | Gpc^3 sr^-1. |
| `base_cosmology.ddl_by_dz_at_z` | `ddl_dz` | implemented/renamed | Analytic for flat LCDM, finite difference for wrappers. |
| `base_cosmology.sample_comoving_volume` | `sample_comoving_volume` | implemented/renamed | Seeded RNG supported. |
| `astropycosmology` | `FlatLambdaCDM` | implemented/renamed | Native Julia, no Astropy dependency. |
| `eps0_astropycosmology` | `Epsilon0Cosmology` | implemented/renamed | Luminosity-distance wrapper. |
| `Xi0_astropycosmology` | `Xi0Cosmology` | implemented/renamed | Luminosity-distance wrapper. |
| `extraD_astropycosmology` | `ExtraDCosmology` | implemented/renamed | Luminosity-distance wrapper. |
| `cM_astropycosmology` | `PlanckMassCosmology` | implemented/renamed | Native integral. |
| `alphalog_astropycosmology` | `AlphaLogCosmology` | implemented/renamed | Luminosity-distance wrapper. |
| `galaxy_MF`, `basic_absM_rate`, `log_powerlaw_absM_rate` | catalog module placeholder | planned | Catalog first-version exclusion. |
| `powerlaw_rate`, `md_rate`, `md_gamma_rate`, `beta_rate`, `beta_rate_line` | `PowerLawRate`, `MadauRate`, `MadauGammaRate`, `BetaRate`, `BetaLineRate` | implemented/renamed | Native structs. |

## conversions.py

| Python API | Julia API | Status | Notes |
| --- | --- | --- | --- |
| `chirp_mass` | `chirp_mass` | implemented | Scalar and array broadcasting. |
| `mass_ratio` | `mass_ratio` | implemented | `m2/m1`. |
| `f_GW_ISCO` | `f_gw_isco` | implemented/renamed | Python-compatible formula. |
| `L2M`, `M2L` | `L2M`, `M2L` | implemented | IAU zero point. |
| `M2m`, `m2M` | `apparent_magnitude`, `absolute_magnitude` | implemented/renamed | Mpc luminosity distance. |
| `source2detector` | `source_to_detector`, `source2detector` | implemented/renamed | Alias retained. |
| `detector2source` | `detector_to_source`, `detector2source` | implemented/renamed | Alias retained. |
| Jacobian helpers | `detector_to_source_jacobian*`, `source_to_detector_jacobian` | implemented/renamed | Used in likelihood. |
| `chi_eff_from_spins`, `chi_p_from_spins`, `cartestianspins2chis` | `chi_eff_from_spins`, `chi_p_from_spins`, `cartesian_spins_to_chis` | implemented/renamed | Core spin conversions. |
| skymap/HEALPix helpers | catalog/skymap future module | planned | Not first-version core. |

## priors.py and wrappers.py

| Python API | Julia API | Status | Notes |
| --- | --- | --- | --- |
| `PowerLaw` | `PowerLaw` | implemented | Normalized density. |
| `BetaDistribution`, `TruncatedBetaDistribution` | same names | implemented | Native `Distributions.jl`. |
| `TruncatedGaussian` | `TruncatedGaussian` | implemented | Native `Distributions.jl`. |
| `PowerLawGaussian` | `PowerLawGaussian` | implemented | Power-law plus peak. |
| `BrokenPowerLaw` | `BrokenPowerLaw` | implemented | Continuous broken law. |
| `PowerLawTwoGaussians` | `PowerLawTwoGaussians` | implemented | Multi-peak mass model. |
| `BrokenPowerLawMultiPeak` | `BrokenPowerLawMultiPeak` | implemented | Broken law plus two peaks. |
| `BrokenPowerLawTripleMultiPeak` | same name | implemented | Broken law plus three peaks. |
| `conditional_2dimpdf` | `ConditionalMassDistribution` | implemented/renamed | Conditional `m2 <= m1`. |
| `paired_2dimpdf` | `PairedMassDistribution` | implemented/renamed | Pairing by mass ratio power. |
| `piecewise_constant_2d_distribution_normalized` | `PiecewiseConstant2D` | implemented/renamed | Triangular checkerboard. |
| low-pass / dip smoothers | `LowpassSmoothedProb`, `SmoothedPlusDipProb` | implemented | Native numerical normalization. |
| spin prior wrappers | `DefaultSpinPrior`, `GaussianSpinPrior` | implemented/merged | Core population spin priors. |
| Python mutable wrapper classes | `ParameterSchema` + model structs | merged | Julia hot path uses vectors and immutable models. |

## rates.py

| Python API | Julia API | Status | Notes |
| --- | --- | --- | --- |
| `CBC_vanilla_rate` | `CBCVanillaRate`, `SimplePowerLawPopulation` | implemented/renamed | Detector `(m1, m2, dL)`. |
| `CBC_rate_m1_q` | `CBCMass1Rate` | implemented/renamed | Detector `(m1, q, dL)`. |
| `CBC_rate_mchirp_q` | `CBCMchirpQRate` | implemented/renamed | Detector `(Mc, q, dL)`. |
| spin variants | compose spin priors with rate models | partial/planned | Core spin priors exist; all Python combinations need more regression fixtures. |
| catalog/EM counterpart rates | placeholder modules | planned | First-version exclusion. |
| stochastic mixed rates | placeholder modules | planned | First-version exclusion. |

## posterior_samples.py and injections.py

| Python API | Julia API | Status | Notes |
| --- | --- | --- | --- |
| `posterior_samples` | `PosteriorSamples` | implemented/renamed | Array-backed columns. |
| `posterior_samples_catalog` | `PosteriorSampleSet` | implemented/renamed | Ordered collection. |
| `injections` | `InjectionSet` | implemented/renamed | Includes `ntotal` and `Tobs`. |
| `update_weights` methods | likelihood internals | merged | Weights are computed by pure functions. |
| pixelization/catalog methods | catalog future module | planned | First-version exclusion. |

## likelihood.py

| Python API | Julia API | Status | Notes |
| --- | --- | --- | --- |
| `hierarchical_likelihood` | `loglikelihood` | implemented/renamed | No `bilby.Likelihood` inheritance. |
| `hierarchical_likelihood_v1` | `loglikelihood` | merged | Deterministic scalar function. |
| `hierarchical_likelihood_noevents` | `no_event_loglikelihood` | implemented/renamed | Upper-limit likelihood. |
| selection correction | `InjectionSet` + diagnostics | implemented | `xi`, `N_expected`, injection ESS. |
| diagnostics | `LikelihoodDiagnostics` | implemented | Structured return type. |
| stochastic likelihood classes | `Stochastic.stochastic_planned` | planned | First-version exclusion. |

## simulation.py

| Python API | Julia API | Status | Notes |
| --- | --- | --- | --- |
| source mass/redshift generation | `simulate_sources` | implemented/renamed | Seeded RNG. |
| SNR helpers | `snr_samples`, `apply_snr_cut` | implemented/renamed | Smoke-test approximation. |
| quick PE generation | `generate_posterior_samples` | implemented/renamed | Toy posteriors. |
| injection generation | `generate_injections` | implemented/renamed | Native injection set. |
| end-to-end mock data | `simulate_population_data` | implemented | Used by tests/examples. |

## catalog.py, stochastic.py, omega_gw.py, utils.py, cupy_pal.py

| Python API | Julia API | Status | Notes |
| --- | --- | --- | --- |
| galaxy catalog / EM counterpart APIs | `Catalog.catalog_planned` | planned | Clear planned error. |
| stochastic background APIs | `Stochastic.stochastic_planned` | planned | Clear planned error. |
| `Omega_GW` helpers | `OmegaGW.omega_gw_planned` | planned | Clear planned error. |
| Condor helper functions | none | excluded | Explicitly excluded. |
| pickle support | none | excluded | Explicitly excluded. |
| `cupy_pal.py` GPU backend switching | none | excluded | Native Julia CPU first; no Python/CuPy bridge. |

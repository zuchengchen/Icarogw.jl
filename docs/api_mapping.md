# Python-To-Julia API Mapping

This document tracks first-version migration status from `../icarogw/icarogw`
to native Julia `Icarogw.jl`.

For the active full-science migration goal, use
[`docs/migration_gap_audit.csv`](migration_gap_audit.csv) as the
machine-readable gap tracker and [`docs/migration_gap_audit.md`](migration_gap_audit.md)
as the human-readable phase plan. Items marked `planned` here are no longer
out of scope for the full-science goal unless the gap audit marks them
`excluded`.

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
| `FlatwCDM_wrap` | `FlatwCDM` | implemented/renamed | Constant-`w` native cosmology. |
| `Flatw0waCDM_wrap` | `Flatw0waCDM` | implemented/renamed | CPL `w0-wa` native cosmology. |
| `eps0_astropycosmology` | `Epsilon0Cosmology` | implemented/renamed | Luminosity-distance wrapper. |
| `Xi0_astropycosmology` | `Xi0Cosmology` | implemented/renamed | Luminosity-distance wrapper. |
| `extraD_astropycosmology` | `ExtraDCosmology` | implemented/renamed | Luminosity-distance wrapper. |
| `cM_astropycosmology` | `PlanckMassCosmology` | implemented/renamed | Native integral. |
| `alphalog_astropycosmology` | `AlphaLogCosmology` | implemented/renamed | Luminosity-distance wrapper. |
| `galaxy_MF`, `basic_absM_rate`, `log_powerlaw_absM_rate` | `GalaxyLuminosityFunction`, `AbstractAbsMagnitudeRate`, `LogPowerLawAbsMagnitudeRate` | implemented/renamed | Dependency-light Schechter and absolute-magnitude rate formulas with fixture coverage; runtime catalog readers consume them, while preprocessing and catalog-aware rates remain separate gaps. |
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
| `cred_interval` | `cred_interval` | implemented | Gaussian symmetric interval probability. |
| `chi_effective_prior_from_aligned_spins` | same name | implemented | Native piecewise density. |
| `chi_effective_prior_from_isotropic_spins` | same name | implemented | Native convolution/quadrature density. |
| `chi_p_prior_from_isotropic_spins` | same name | implemented | Native maximum-distribution density. |
| `joint_prior_from_isotropic_spins`, `chi_p_prior_given_chi_eff_q` | same names | implemented | RNG-explicit Monte Carlo and weighted KDE helpers. |
| `chi_eff_from_spins`, `chi_p_from_spins`, `cartestianspins2chis` | `chi_eff_from_spins`, `chi_p_from_spins`, `cartesian_spins_to_chis` | implemented/renamed | Core spin conversions. |
| `radec2skymap`, `radec2indeces`, `indices2radec` | same names plus `radec2indices` | implemented/renamed | Backed by `Healpix.jl`; Julia indices are 1-based by default with `zero_based=true` for Python compatibility. |
| `ligo_skymap` | `LigoSkyMap`, `ligo_skymap` | partial | FITSIO/Healpix-backed multi-order `UNIQ` skymap reader with distance layers, 3D posterior/likelihood, and sampling; catalog/EM integration remains pending. |

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
| `conditional_2dimz_pdf` | `RedshiftConditionalMassDistribution` | implemented/renamed | Redshift-dependent primary `p(m1|z)` with conditional secondary. |
| `paired_2dimpdf` | `PairedMassDistribution`, `GeneralPairedMassDistribution` | implemented/renamed | Pairing by mass-ratio power or a custom pairing function. |
| `piecewise_constant_2d_distribution_normalized` | `PiecewiseConstant2D` | implemented/renamed | Triangular checkerboard. |
| `m1m2_paired_massratio_dip`, `m1m2_paired_massratio_dip_general`, `m1m2_paired_massratio_bpl_dip_farah_2022` | `paired_massratio_dip`, `paired_massratio_dip_general`, `paired_massratio_bpl_dip_farah_2022` | implemented/renamed | Dependency-free dip/Farah paired mass wrapper constructors with Python reference fixtures. |
| `m1m2_paired_massratio_bplmulti_dip`, `m1m2_paired_bpl_triplepeak_dip`, `m1m2_paired_massratio_bplmulti_dip_conditioned` | `paired_massratio_bplmulti_dip`, `paired_bpl_triplepeak_dip`, `paired_massratio_bplmulti_dip_conditioned` | implemented/renamed | Multi-peak dip paired wrappers with fixture coverage; triple-peak fixture avoids Python's inconsistent `lambda_2` log-PDF branch. |
| `massprior_BinModel2d` | `bin_model_2d` | implemented/renamed | Convenience constructor around `PiecewiseConstant2D`. |
| `_lowpass_filter`, `_highpass_filter`, `_notch_filter`, `_mixed_linear_function`, `_mixed_double_sigmoid_function` | `lowpass_filter`, `highpass_filter`, `notch_filter`, `mixed_linear_function`, `mixed_double_sigmoid_function` | implemented/renamed | Helper-level formulas covered by fixture. |
| low-pass / dip smoothers | `LowpassSmoothedProb`, `LowpassSmoothedProbEvolving`, `SmoothedPlusDipProb` | implemented | Evolving smoother follows Python's fixed-grid normalization; non-evolving smoother keeps native numerical normalization. |
| `Bivariate2DGaussian` | `Bivariate2DGaussian` | implemented | Truncated marginal plus conditional Gaussian. |
| `absL_PL_inM` | `AbsLuminosityPowerLawInMagnitude`, `absL_PL_inM` | implemented/renamed | Luminosity power law represented in absolute-magnitude space. |
| `PowerLawStationary`, `PowerLawLinear`, `GaussianStationary`, `GaussianLinear` | same names | implemented | Redshift-stationary and redshift-linear mass components. |
| `PowerLaw_PowerLaw`, `PowerLaw_PowerLaw_PowerLaw`, `PowerLaw_PowerLaw_Gaussian` | `MixtureMassPrior` with stationary components | implemented/merged | Julia-native composable mixture abstraction with fixture coverage. |
| redshift-linear mixture wrappers | `RedshiftMixtureMassPrior` with stationary/linear components | implemented/merged | Weight functions make Python `mix_z0/mix_z1` wrapper families explicit and fixture-backed. |
| spin prior wrappers | `DefaultSpinPrior`, `GaussianComponentSpinPrior`, `EvolvingGaussianSpinPrior`, `BetaWindowGaussianSpinPrior`, `BetaWindowBetaSpinPrior`, `PSEOBGaussianPrior`, `ECOTotallyReflectiveSpinPrior`, `GaussianSpinPrior` | implemented/merged | Component-spin, mass-dependent, pSEOB, ECO, and effective-spin priors are native structs with fixture coverage. |
| Python mutable wrapper classes | `ParameterSchema` + model structs | merged | Julia hot path uses vectors and immutable models. |

## rates.py

| Python API | Julia API | Status | Notes |
| --- | --- | --- | --- |
| `CBC_vanilla_rate` | `CBCVanillaRate`, `SimplePowerLawPopulation` | implemented/renamed | Detector `(m1, m2, dL)`. |
| `CBC_rate_m1_q` | `CBCMass1Rate` | implemented/renamed | Detector `(m1, q, dL)`. |
| `CBC_rate_mchirp_q` | `CBCMchirpQRate` | implemented/renamed | Detector `(Mc, q, dL)`. |
| `CBC_rate_m_given_redshift` | `CBCSingleMassRate` | implemented/renamed | Detector `(m, dL)` with optional redshift-dependent mass prior. |
| `CBC_rate_total_mass_q` | `CBCTotalMassQRate` | implemented/renamed | Detector `(Mtot, q, dL)`. |
| `CBC_rate_m1_given_redshift_q` | `CBCRedshiftPrimaryQRate` | implemented/renamed | Detector `(m1, q, dL)` with `p(m1|z)`. |
| `CBC_rate_m1_given_redshift_m2` | `CBCVanillaRate` with `RedshiftConditionalMassDistribution` | implemented/merged | Detector `(m1, m2, dL)` with `p(m1|z)p(m2|m1)`. |
| `CBC_mixte_pop_rate` | `MixtureRate` | implemented/renamed | Convex logaddexp mixture of two compatible rate models. |
| spin variants | `SpinWeightedRate(base, spin_prior)` | implemented/merged | Composes component-spin, mass-dependent spin, pSEOB/ECO, or effective-spin priors with first-version CBC rate models; mass-dependent spin priors use explicit source-mass columns. |
| `CBC_vanilla_rate_pseob` | `SpinWeightedRate(base, PSEOBGaussianPrior)` | partial/merged | Standard pSEOB weighting is supported; Python `_dummy` injection asymmetry is not yet a separate likelihood contract. |
| `CBC_catalog_vanilla_rate`, `CBC_catalog_vanilla_rate_skymap` | `CBCCatalogVanillaRate`, `CBCCatalogSkyMapRate` | implemented/renamed | Catalog-aware CBC rates consume `:sky_indices`, runtime catalog interpolants, and Python-compatible posterior/injection completeness behavior. |
| EM counterpart rates | placeholder modules | planned | Counterpart-specific rate wrappers remain a separate catalog/EM integration phase. |
| stochastic mixed rates | placeholder modules | planned | First-version exclusion. |

## posterior_samples.py and injections.py

| Python API | Julia API | Status | Notes |
| --- | --- | --- | --- |
| `posterior_samples` | `PosteriorSamples` | implemented/renamed | Array-backed columns. |
| `posterior_samples_catalog` | `PosteriorSampleSet` | implemented/renamed | Ordered collection. |
| `posterior_samples_catalog.build_parallel_posterior` | `build_parallel_posterior`, `ParallelPosterior` | implemented/renamed | RNG-explicit matrix workspace with fill mask and per-event sample counts. |
| `injections` | `InjectionSet` | implemented/renamed | Includes `ntotal` and `Tobs`. |
| `update_weights` methods | likelihood internals | merged | Weights are computed by pure functions. |
| `effective_injections_number`, PE effective number | `effective_sample_size` | implemented/renamed | Works from log weights or model+container. |
| `expected_number_detections` | `expected_number_detections` | implemented | Uses injection pseudo-rate convention. |
| `update_cut`, `reweight_PE`, `return_reweighted_injections` | `subset_injections`, `subset_posterior_samples`, `reweight_posterior_samples`, `reweight_injections` | implemented/renamed | RNG-explicit pure helpers. |
| `posterior_samples.add_counterpart` | `add_counterpart` | implemented/renamed | Dependency-light `z_EM` column attachment; sky-direction filtering can compose with the pixelized `:sky_indices` column. |
| `pixelize`, `pixelize_with_catalog` | `pixelize`, `pixelize_with_catalog` | implemented/renamed | Pure helpers return new posterior/injection containers with `:sky_indices` from HEALPix or catalog NUNIQ/MOC lookup. |

## likelihood.py

| Python API | Julia API | Status | Notes |
| --- | --- | --- | --- |
| `hierarchical_likelihood` | `loglikelihood` | implemented/renamed | No `bilby.Likelihood` inheritance. |
| `hierarchical_likelihood_v1` | `loglikelihood` | merged | Deterministic scalar function. |
| `hierarchical_likelihood_noevents` | `no_event_loglikelihood` | implemented/renamed | Upper-limit likelihood. |
| selection correction | `InjectionSet` + diagnostics | implemented | `xi`, `N_expected`, injection ESS. |
| diagnostics | `LikelihoodDiagnostics` | implemented | Structured return type. |
| stochastic likelihood classes | `StochasticData`, `read_stochastic_csv`, `read_stochastic_hdf5`, `stochastic_loglikelihood`, `joint_loglikelihood` | partial | Gaussian stochastic-only, simple stochastic CSV/HDF5 readers, and vanilla CBC+stochastic helpers are implemented; catalog/EM mixed stochastic likelihoods and richer covariance/data-product APIs remain pending. |

## simulation.py

| Python API | Julia API | Status | Notes |
| --- | --- | --- | --- |
| source mass/redshift generation | `simulate_sources` | implemented/renamed | Seeded RNG. |
| `chirp_mass_det`, `f_GW`, `z_to_dl`, `dl_to_z`, `dVc_dz` | `chirp_mass_detector`, `f_gw`, `z_to_dl`, `dl_to_z`, `dvc_dz_fullsky` | implemented/renamed | Deterministic formulas covered by fixture. |
| SNR helpers | `snr_samples`, `snr_samples_source`, `snr_samples_detector`, `snr_samples_flat`, `apply_snr_cut`, `snr_and_freq_cut`, `snr_cut_flat` | implemented/renamed | Smoke-test approximation; random draws are RNG-explicit. |
| measurement noise and quick likelihood | `chirp_mass_noise`, `mass_ratio_noise`, `theta_noise`, `noise`, `likelihood_evaluation` | implemented/renamed | Formula fixture covers deterministic likelihood factors. |
| quick PE generation | `generate_posterior_samples` | implemented/renamed | Toy posteriors. |
| `generate_mass_inj`, `generate_single_mass_inj`, `generate_dL_inj*`, `injection_set_generator` | same names plus `generate_injections` | implemented/renamed | RNG-explicit mass/distance proposal draws and Python-style injection-set generator; returns an `InjectionSet` for native likelihoods. |
| injection generation | `generate_injections` | implemented/renamed | Native population-model injection set. |
| end-to-end mock data | `simulate_population_data` | implemented | Used by tests/examples. |
| `quick_data_preparation`, `PE_quick_generation_samples` | `quick_data_preparation`, `pe_quick_generation_samples`, `PE_quick_generation_samples` | implemented/renamed | RNG-explicit native quick-prep and PE resampling workflow; `Ngen` controls proposal size. |

## catalog.py, stochastic.py, omega_gw.py, utils.py, cupy_pal.py

| Python API | Julia API | Status | Notes |
| --- | --- | --- | --- |
| `kcorr`, `kcorr_dep` | `KCorrection`, `DeprecatedKCorrection` | implemented/renamed | Modern and legacy dependency-light K-correction formulas with Python fixture coverage. |
| `galaxy_MF_dep` | `LegacyGalaxyLuminosityFunction`, `galaxy_MF_dep` | implemented/renamed | Legacy W1/K/bJ Schechter helper with Python fixture coverage; `GalaxyLuminosityFunction` remains preferred for new code. |
| `EM_likelihood_prior_differential_volume` | `em_likelihood_prior_differential_volume`, `EM_likelihood_prior_differential_volume` | implemented/renamed | Uniform, gaussian, and gaussian-without-comoving-volume redshift helper with fixture coverage. |
| `icarogw_catalog`, `gwcosmo_catalog` | `IcarogwCatalog`, `GwcosmoCatalog` | implemented/renamed | Runtime HDF5 readers, NUNIQ/HEALPix row lookup, magnitude thresholds, effective galaxy interpolants, and `make_me_empty!` compatibility helpers. Pixelated catalog builders remain a separate gap. |
| `galaxy_catalog` | `GalaxyCatalog`, `galaxy_catalog`, `create_hdf5`, `load_hdf5`, `calculate_mthr!`, `return_counts_map` | partial/renamed | Single-file HDF5 runtime compatibility is implemented for catalog columns, Python-style zero-based stored `sky_indices`, counts maps, magnitude-threshold maps including `"empty"`, and effective galaxy interpolants. Long-running pixelated preprocessing builders remain separate gaps. |
| stochastic background APIs | `dedf`, `precompute_omega_weights`, `spectral_siren_omega_gw`, `StochasticData`, `read_stochastic_csv`, `read_stochastic_hdf5`, `stochastic_loglikelihood` | implemented/renamed | Energy spectrum, omega weights, vanilla spectral-siren helper, Gaussian stochastic likelihood, and simple freqs/Cf/sigma2s readers are implemented. |
| `Omega_GW` helpers | `dedf`, `precompute_omega_weights`, `spectral_siren_omega_gw`, `joint_loglikelihood` | implemented/merged | Duplicate Python `stochastic.py`/`omega_gw.py` formulas are unified in Julia with a vanilla CBC+stochastic helper. |
| `utils.check_posterior_samples_and_prior` | `check_posterior_samples_and_prior` | implemented | Non-Condor validation helper. |
| `cupy_pal.check_bounds_1D`, `check_bounds_2D` | `check_bounds_1d`, `check_bounds_2d` | implemented/renamed | CPU array semantics only. |
| Condor helper functions | none | excluded | Explicitly excluded. |
| pickle support | none | excluded | Explicitly excluded. |
| `cupy_pal.py` GPU backend switching | none | excluded | Native Julia CPU first; no Python/CuPy bridge. |

using Icarogw
using CSV
using DataFrames
using Random
using Test

function _tiny_population_data(c=FlatLambdaCDM(zmax=2))
    z1, z2 = 0.2, 0.35
    m1a, m2a, dla = source_to_detector(30.0, 20.0, z1, c)
    m1b, m2b, dlb = source_to_detector(32.0, 18.0, z2, c)
    ps1 = PosteriorSamples((mass_1=[m1a, 1.02m1a, 0.98m1a],
        mass_2=[m2a, 1.01m2a, 0.99m2a],
        luminosity_distance=[dla, 1.01dla, 0.99dla],
        prior=fill(1.0, 3)); event_name=:event1)
    ps2 = PosteriorSamples((mass_1=[m1b, 1.01m1b, 0.99m1b],
        mass_2=[m2b, 0.99m2b, 1.01m2b],
        luminosity_distance=[dlb, 1.01dlb, 0.99dlb],
        prior=fill(1.0, 3)); event_name=:event2)
    inj = InjectionSet((mass_1=[m1a, m1b, 1.04m1a, 0.97m1b],
        mass_2=[m2a, m2b, 0.96m2a, 1.02m2b],
        luminosity_distance=[dla, dlb, 1.02dla, 0.98dlb],
        prior=fill(1.0, 4)); ntotal=8, Tobs=2.0)
    return PopulationData(PosteriorSampleSet(ps1, ps2), inj)
end

@testset "cosmology and conversions" begin
    c = FlatLambdaCDM(H0=67.7, Om0=0.308, zmax=5)
    z = 0.5
    dl = luminosity_distance(c, z)
    @test dl > 0
    @test redshift_at_luminosity_distance(c, dl) ≈ z rtol=1e-7
    @test dvc_dz(c, z) > 0
    @test ddl_dz(c, z) > 0
    @test luminosity_distance(FlatwCDM(H0=67.7, Om0=0.308, w0=-1, zmax=5), z) ≈ dl rtol=1e-10
    @test luminosity_distance(Flatw0waCDM(H0=67.7, Om0=0.308, w0=-1, wa=0, zmax=5), z) ≈ dl rtol=1e-10
    @test cred_interval(1.0) ≈ 0.6826894921370859 rtol=1e-12
    @test chirp_mass(30.0, 20.0) ≈ (30 * 20)^(3 / 5) / 50^(1 / 5)
    @test mass_ratio(30.0, 20.0) ≈ 2 / 3
    @test M2L(L2M(3.0128e28)) ≈ 3.0128e28 rtol=1e-6
    @test chi_effective_prior_from_aligned_spins(0.8, 1.0, 0.0) > 0
    @test chi_effective_prior_from_isotropic_spins(0.8, 1.0, 0.0) > 0
    @test chi_p_prior_from_isotropic_spins(0.8, 1.0, 0.3) > 0
end

@testset "reference fixtures" begin
    refdir = joinpath(@__DIR__, "reference")

    cosmology_models = Dict(
        "flatlcdm" => FlatLambdaCDM(H0=67.7, Om0=0.308, zmax=5),
        "flatw" => FlatwCDM(H0=67.7, Om0=0.308, w0=-0.8, zmax=5),
        "flatw0wa" => Flatw0waCDM(H0=67.7, Om0=0.308, w0=-0.9, wa=0.2, zmax=5),
    )
    cosmo_ref = CSV.File(joinpath(refdir, "reference_cosmology_core.csv")) |> DataFrame
    for row in eachrow(cosmo_ref)
        c = cosmology_models[row.model]
        @test luminosity_distance(c, row.z) ≈ row.luminosity_distance rtol=3e-7
        @test comoving_volume(c, row.z) ≈ row.comoving_volume rtol=3e-6
        @test dvc_dz_dOmega(c, row.z) ≈ row.dvc_dz_dOmega rtol=4e-6
        @test ddl_dz(c, row.z) ≈ row.ddl_dz rtol=4e-7
        @test redshift_at_luminosity_distance(c, row.luminosity_distance) ≈ row.dl2z atol=2e-7
    end

    conv_ref = only(CSV.File(joinpath(refdir, "reference_conversions_core.csv")) |> DataFrame |> eachrow)
    @test chirp_mass(conv_ref.m1, conv_ref.m2) ≈ conv_ref.chirp_mass rtol=1e-14
    @test mass_ratio(conv_ref.m1, conv_ref.m2) ≈ conv_ref.mass_ratio rtol=1e-14
    @test f_gw_isco(conv_ref.m1, conv_ref.m2) ≈ conv_ref.f_gw_isco rtol=1e-14
    @test L2M(conv_ref.L) ≈ conv_ref.L2M rtol=1e-14 atol=1e-14
    @test M2L(conv_ref.M) ≈ conv_ref.M2L rtol=1e-14
    @test chi_eff_from_spins(conv_ref.chi1, conv_ref.chi2, conv_ref.cos1, conv_ref.cos2, conv_ref.q) ≈ conv_ref.chi_eff rtol=1e-14
    @test chi_p_from_spins(conv_ref.chi1, conv_ref.chi2, conv_ref.cos1, conv_ref.cos2, conv_ref.q) ≈ conv_ref.chi_p rtol=1e-14

    spin_ref = CSV.File(joinpath(refdir, "reference_spin_core.csv")) |> DataFrame
    for row in eachrow(spin_ref)
        @test chi_effective_prior_from_aligned_spins(row.q, row.amax, row.x_eff) ≈ row.aligned rtol=1e-14
        @test chi_effective_prior_from_isotropic_spins(row.q, row.amax, row.x_eff) ≈ row.isotropic rtol=4e-6
        @test chi_p_prior_from_isotropic_spins(row.q, row.amax, row.x_p) ≈ row.chi_p rtol=1e-14
    end

    prior_models = Dict{String,Any}(
        "PowerLaw" => PowerLaw(5.0, 80.0, -2.0),
        "TruncatedGaussian" => TruncatedGaussian(30.0, 5.0, 5.0, 80.0),
        "PowerLawGaussian" => PowerLawGaussian(5.0, 80.0, -2.0, 0.1, 35.0, 4.0, 5.0, 55.0),
        "BrokenPowerLaw" => BrokenPowerLaw(5.0, 80.0, -1.5, -3.0, 0.4),
        "BetaDistribution" => BetaDistribution(2.0, 5.0),
        "TruncatedBetaDistribution" => TruncatedBetaDistribution(2.0, 5.0, 0.8),
    )
    priors_ref = CSV.File(joinpath(refdir, "reference_priors_core.csv")) |> DataFrame
    for row in eachrow(priors_ref)
        prior = prior_models[row.prior]
        @test logpdf(prior, row.x) ≈ row.logpdf rtol=1e-13 atol=1e-13
        @test cdf(prior, row.x) ≈ row.cdf rtol=1e-12 atol=1e-14
    end
end

@testset "priors and schema" begin
    p = PowerLaw(5, 80, -2)
    @test isfinite(logpdf(p, 10.0))
    @test logpdf(p, 1.0) == -Inf
    @test 0 < cdf(p, 10.0) < 1

    g = TruncatedGaussian(30, 5, 5, 80)
    mix = PowerLawGaussian(5, 80, -2, 0.1, 35, 4, 5, 55)
    @test isfinite(logpdf(g, 30.0))
    @test isfinite(logpdf(mix, 35.0))

    cm = ConditionalMassDistribution(PowerLaw(5, 80, -2), PowerLaw(5, 80, 1))
    @test isfinite(logpdf(cm, 30.0, 20.0))
    @test logpdf(cm, 20.0, 30.0) == -Inf
    plz = PowerLawLinear(2.0, 0.2, 5.0, 0.5, 80.0, 1.0)
    @test isfinite(logpdf(plz, 30.0, 0.4))
    gz = GaussianLinear(30.0, 2.0, 5.0, 0.1, 5.0)
    @test isfinite(logpdf(gz, 31.0, 0.5))
    mixz = MixtureMassPrior((PowerLawStationary(2.0, 5.0, 80.0), gz), [0.7, 0.3])
    @test isfinite(logpdf(mixz, 30.0, 0.2))

    schema = parameter_schema(SimplePowerLawPopulation)
    theta = prior_transform(schema, fill(0.5, length(schema)))
    named = unpack(schema, theta)
    @test pack(schema, named) == theta
end

@testset "data containers and likelihood" begin
    rng = MersenneTwister(42)
    truth = SimplePowerLawPopulation()
    data = simulate_population_data(rng, truth; nevents=2, nsamples=64, ndetected=64, ntotal=500, zmax=0.5)
    validate(data)
    schema = parameter_schema(SimplePowerLawPopulation)
    theta = pack(schema, (
        alpha=2.0, beta=1.0, mmin=5.0, mmax=80.0,
        gamma=0.0, R0=25.0, H0=67.7, Om0=0.308,
    ))
    logl = loglikelihood(SimplePowerLawPopulation, data, theta)
    @test isfinite(logl)
    diag = likelihood_diagnostics(SimplePowerLawPopulation, data, theta)
    @test length(diag.per_event_neff) == 2
    @test diag.injection_neff > 0
    @test diag.xi > 0
    @test diag.N_expected > 0
    @test isfinite(no_event_loglikelihood(materialize(SimplePowerLawPopulation, theta), data.injections))

    batch = hcat(theta, theta)
    vals = loglikelihood_batch(SimplePowerLawPopulation, data, batch)
    @test length(vals) == 2
    @test vals[1] ≈ vals[2]

    tiny = _tiny_population_data()
    model = SimplePowerLawPopulation(cosmology=FlatLambdaCDM(zmax=2), R0=10.0)
    poisson_logl = loglikelihood(model, tiny; options=LikelihoodOptions(poisson=true))
    no_poisson_logl = loglikelihood(model, tiny; options=LikelihoodOptions(poisson=false))
    shape_logl = loglikelihood(model, tiny; options=LikelihoodOptions(shape_only=true))
    diag_tiny = likelihood_diagnostics(model, tiny)
    @test poisson_logl ≈ no_poisson_logl - diag_tiny.N_expected + length(tiny.posteriors) * log(tiny.injections.Tobs)
    @test shape_logl ≈ no_poisson_logl - length(tiny.posteriors) * log(diag_tiny.xi)
    @test no_event_loglikelihood(model, tiny.injections) ≈ -diag_tiny.N_expected
    @test likelihood_diagnostics(model, tiny; options=LikelihoodOptions(neff_event_min=100)).accepted == false
    @test loglikelihood(model, tiny; options=LikelihoodOptions(neff_injection_min=100)) == -Inf
    @test loglikelihood_batch(SimplePowerLawPopulation, data, batch; parallel=true) ≈ vals
end

@testset "io and validation" begin
    ps = PosteriorSamples((mass_1=[30.0, 31.0], mass_2=[20.0, 19.5],
        luminosity_distance=[1000.0, 1100.0], prior=[1.0, 2.0]); event_name=:gw1)
    inj = InjectionSet((mass_1=[32.0, 33.0], mass_2=[21.0, 20.5],
        luminosity_distance=[1200.0, 1300.0], prior=[1.5, 1.6]); ntotal=4, Tobs=1.5)
    mktempdir() do dir
        ps_csv = joinpath(dir, "posterior.csv")
        inj_csv = joinpath(dir, "injections.csv")
        CSV.write(ps_csv, DataFrame(mass_1=collect(column(ps, :mass_1)), mass_2=collect(column(ps, :mass_2)),
            luminosity_distance=collect(column(ps, :luminosity_distance)), prior=ps.prior))
        CSV.write(inj_csv, DataFrame(mass_1=collect(column(inj, :mass_1)), mass_2=collect(column(inj, :mass_2)),
            luminosity_distance=collect(column(inj, :luminosity_distance)), prior=inj.prior))
        ps_from_csv = read_posterior_csv(ps_csv; event_name=:gw1)
        inj_from_csv = read_injections_csv(inj_csv; ntotal=4, Tobs=1.5)
        @test ps_from_csv.names == ps.names
        @test Matrix(ps_from_csv.values) ≈ Matrix(ps.values)
        @test ps_from_csv.prior ≈ ps.prior
        @test inj_from_csv.names == inj.names
        @test Matrix(inj_from_csv.values) ≈ Matrix(inj.values)
        @test inj_from_csv.prior ≈ inj.prior
        @test inj_from_csv.Tobs == 1.5

        ps_h5 = joinpath(dir, "posterior.h5")
        inj_h5 = joinpath(dir, "injections.h5")
        write_hdf5(ps_h5, ps)
        write_hdf5(inj_h5, inj)
        ps_from_h5 = read_posterior_hdf5(ps_h5)
        inj_from_h5 = read_injections_hdf5(inj_h5)
        @test ps_from_h5.event_name == :gw1
        @test ps_from_h5.names == ps.names
        @test ps_from_h5.values ≈ ps.values
        @test ps_from_h5.prior ≈ ps.prior
        @test inj_from_h5.names == inj.names
        @test inj_from_h5.values ≈ inj.values
        @test inj_from_h5.prior ≈ inj.prior
        @test inj_from_h5.ntotal == 4
        @test inj_from_h5.Tobs == 1.5
    end

    @test_throws ArgumentError PosteriorSamples((mass_1=[30.0], prior=[0.0]))
    @test_throws ArgumentError PosteriorSamples((mass_1=[30.0], luminosity_distance=[Inf], prior=[1.0]))
    @test_throws ArgumentError InjectionSet((mass_1=[30.0, 31.0], prior=[1.0]); ntotal=2)
    @test_throws ArgumentError loglikelihood(CBCMass1Rate(FlatLambdaCDM(zmax=2), PowerLaw(5.0, 100.0, -2.0),
        PowerLaw(0.1, 1.0, 0.0), PowerLawRate(0.0)), PopulationData(PosteriorSampleSet(ps), inj))
end

@testset "simulation sanity" begin
    rng = MersenneTwister(123)
    model = SimplePowerLawPopulation(cosmology=FlatLambdaCDM(zmax=1))
    sources = simulate_sources(rng, 16, model; zmax=0.4)
    @test length(sources.mass_1) == 16
    @test all(>(0), sources.mass_1)
    @test all(>(0), sources.mass_2)
    @test all(sources.mass_1 .>= sources.mass_2)
    @test all(0 .< sources.redshift .<= 0.4)
    snr = snr_samples(rng, sources.mass_1, sources.mass_2, sources.luminosity_distance)
    mask = apply_snr_cut(sources, snr.rho_obs; snr_threshold=0.0)
    @test mask == (f_gw_isco.(sources.mass_1, sources.mass_2) .>= 15.0)
    post = generate_posterior_samples(rng, (mass_1=40.0, mass_2=25.0, luminosity_distance=1500.0); nsamples=12, event_name=:mock)
    @test length(post.prior) == 12
    @test post.event_name == :mock
end

@testset "alternate rate coordinates" begin
    c = FlatLambdaCDM(zmax=2)
    z = 0.2
    m1d, m2d, dl = source_to_detector(30.0, 20.0, z, c)
    q = mass_ratio(m1d, m2d)
    mc = chirp_mass(m1d, m2d)
    mt = m1d + m2d
    prior = 1.0
    mass = PowerLaw(5.0, 100.0, -2.0)
    qprior = PowerLaw(0.05, 1.0, 0.0)
    rate = PowerLawRate(0.0)

    @test isfinite(log_event_rate(CBCMass1Rate(c, mass, qprior, rate; R0=1), m1d, q, dl, prior))
    @test isfinite(log_event_rate(CBCMchirpQRate(c, mass, qprior, rate; R0=1), mc, q, dl, prior))
    @test isfinite(log_event_rate(CBCSingleMassRate(c, mass, rate; R0=1), m1d, dl, prior))
    @test isfinite(log_event_rate(CBCTotalMassQRate(c, mass, qprior, rate; R0=1), mt, q, dl, prior))
    redshift_mass = PowerLawLinear(2.0, 0.0, 5.0, 0.0, 100.0, 0.0)
    @test isfinite(log_event_rate(CBCRedshiftPrimaryQRate(c, redshift_mass, qprior, rate; R0=1), m1d, q, dl, prior))

    ps = PosteriorSamples((mass_1=[m1d], mass_ratio=[q], luminosity_distance=[dl], prior=[1.0]))
    inj = InjectionSet((mass_1=[m1d], mass_ratio=[q], luminosity_distance=[dl], prior=[1.0]); ntotal=10, Tobs=1)
    data = PopulationData(PosteriorSampleSet(ps), inj)
    @test isfinite(loglikelihood(CBCMass1Rate(c, mass, qprior, rate; R0=1), data))
end

@testset "planned placeholders" begin
    @test_throws ErrorException Icarogw.Catalog.catalog_planned()
    @test_throws ErrorException Icarogw.Stochastic.stochastic_planned()
    @test_throws ErrorException Icarogw.OmegaGW.omega_gw_planned()
end

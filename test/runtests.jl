using Icarogw
using Random
using Test

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

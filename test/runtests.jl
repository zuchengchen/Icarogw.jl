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
    @test chirp_mass(30.0, 20.0) ≈ (30 * 20)^(3 / 5) / 50^(1 / 5)
    @test mass_ratio(30.0, 20.0) ≈ 2 / 3
    @test M2L(L2M(3.0128e28)) ≈ 3.0128e28 rtol=1e-6
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

@testset "planned placeholders" begin
    @test_throws ErrorException Icarogw.Catalog.catalog_planned()
    @test_throws ErrorException Icarogw.Stochastic.stochastic_planned()
    @test_throws ErrorException Icarogw.OmegaGW.omega_gw_planned()
end

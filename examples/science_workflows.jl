using HDF5
using Icarogw
using Random

rng = MersenneTwister(20260701)

cosmology = FlatLambdaCDM(H0=67.7, Om0=0.308, zmax=2.0)
mass_model = ConditionalMassDistribution(PowerLaw(5.0, 80.0, -2.0), PowerLaw(5.0, 80.0, 1.0))
redshift_rate = PowerLawRate(0.0)

function toy_icarogw_catalog(path, cosmology)
    nside = 1
    npix = 12 * nside^2
    level = healpix_nside_to_level(nside)
    uniq = [level_ipix_to_uniq(level, ipix) for ipix in 0:(npix - 1)]
    z_grid = [0.05, 0.15, 0.30]
    dngal = [25.0 * iz + pix for iz in eachindex(z_grid), pix in 1:npix]
    bg = dngal ./ 8

    h5open(path, "w") do h
        group = create_group(h, "K")
        write(group, "mthr_moc_map", fill(20.0, npix))
        write(group, "uniq_moc_map", uniq)
        write(group, "z_grid", z_grid)
        subgroup = create_group(group, "weighted")
        attrs(subgroup)["band"] = "K-glade+"
        attrs(subgroup)["epsilon"] = 0.8
        write(subgroup, "vals_interpolant", dngal)
        write(subgroup, "bg_vals_interpolant", bg)
    end

    return IcarogwCatalog(path, "K", "weighted"; cosmology)
end

mktempdir() do dir
    catalog = toy_icarogw_catalog(joinpath(dir, "toy_icarogw_catalog.h5"), cosmology)
    sky_row = get_NUNIQ_pixel(catalog, 0.2, 0.1)
    event_z = 0.15
    m1d, m2d, dl = source_to_detector(30.0, 20.0, event_z, cosmology)

    catalog_rate = CBCCatalogVanillaRate(catalog, cosmology, mass_model, redshift_rate; Rgal=2.0)
    catalog_ps = PosteriorSamples((mass_1=[m1d, 1.01m1d], mass_2=[m2d, 0.99m2d],
        luminosity_distance=[dl, 1.01dl], sky_indices=[sky_row, sky_row], prior=ones(2)))
    catalog_inj = InjectionSet((mass_1=[m1d, 1.02m1d], mass_2=[m2d, 0.98m2d],
        luminosity_distance=[dl, 1.02dl], sky_indices=[sky_row, sky_row], prior=ones(2));
        ntotal=8, Tobs=1.0)
    dark_siren_logl = loglikelihood(catalog_rate, PopulationData(PosteriorSampleSet(catalog_ps), catalog_inj))
    println("catalog dark-siren loglikelihood = ", dark_siren_logl)

    em_z = [0.12, 0.13, 0.14, 0.15]
    em_dl = luminosity_distance(cosmology, em_z)
    em_rate = CBCVanillaEMCounterpartRate(cosmology, mass_model, redshift_rate; R0=2.0)
    em_ps = PosteriorSamples((mass_1=fill(m1d, length(em_z)), mass_2=fill(m2d, length(em_z)),
        luminosity_distance=em_dl, z_EM=em_z .+ 0.001, prior=ones(length(em_z))); event_name=:event1)
    em_inj = InjectionSet((mass_1=fill(m1d, length(em_z)), mass_2=fill(m2d, length(em_z)),
        luminosity_distance=em_dl, prior=ones(length(em_z))); ntotal=16, Tobs=1.0)
    bright_siren_logl = loglikelihood(em_rate, PopulationData(PosteriorSampleSet(em_ps), em_inj))
    println("vanilla EM bright-siren loglikelihood = ", bright_siren_logl)

    skymap_nside = 1
    skymap_level = healpix_nside_to_level(skymap_nside)
    skymap_uniq = [level_ipix_to_uniq(skymap_level, ipix) for ipix in 0:(12 * skymap_nside^2 - 1)]
    skymap = LigoSkyMap(skymap_uniq, fill(inv(4pi), length(skymap_uniq)),
        fill(luminosity_distance(cosmology, 0.13), length(skymap_uniq)), fill(80.0, length(skymap_uniq)))
    sky_em_rate = CBCLowLatencySkyMapEMCounterpartRate(cosmology, redshift_rate, [skymap]; R0=2.0)
    sky_em_ps = PosteriorSamples((z_EM=em_z, right_ascension=fill(0.2, length(em_z)),
        declination=fill(0.1, length(em_z)), prior=ones(length(em_z))); event_name=:event1)
    sky_em_inj = InjectionSet((luminosity_distance=em_dl, prior=ones(length(em_z))); ntotal=16, Tobs=1.0)
    skymap_em_logl = loglikelihood(sky_em_rate, PopulationData(PosteriorSampleSet(sky_em_ps), sky_em_inj))
    println("skymap EM bright-siren loglikelihood = ", skymap_em_logl)

    stochastic_model = SimplePowerLawPopulation(cosmology=FlatLambdaCDM(H0=67.7, Om0=0.308, zmax=10.0),
        mass=mass_model, redshift_rate=redshift_rate, R0=25.0)
    freqs = [20.0, 40.0, 80.0]
    omega_weights = precompute_omega_weights(MersenneTwister(9), freqs; tmp_min=5.0, tmp_max=12.0, n=16, pn=false)
    omega = spectral_siren_omega_gw(stochastic_model, omega_weights)
    stochastic_data = StochasticData(freqs, omega .+ [1e-12, -1e-12, 2e-12],
        fill(4e-24, length(freqs)); reference_H0=stochastic_model.cosmology.H0)
    stochastic_logl = stochastic_loglikelihood(stochastic_model, omega_weights, stochastic_data)
    println("stochastic-only loglikelihood = ", stochastic_logl)

    population_data = simulate_population_data(rng, stochastic_model; nevents=1, nsamples=32,
        ndetected=48, ntotal=400, zmax=0.5)
    joint_logl = joint_loglikelihood(stochastic_model, population_data, omega_weights, stochastic_data)
    println("joint CBC+stochastic loglikelihood = ", joint_logl)

    problem = dynesty_problem(SimplePowerLawPopulation, population_data; zmax=1.0)
    transformed = problem.prior_transform(fill(0.5, problem.ndim))
    length(transformed) == problem.ndim || error("Dynesty prior transform returned the wrong dimension")
    theta = pack(problem.schema, (alpha=2.0, beta=1.0, mmin=5.0, mmax=80.0,
        gamma=0.0, R0=25.0, H0=67.7, Om0=0.308))
    dynesty_logl = problem.loglikelihood(theta)
    println("Dynesty-compatible closure loglikelihood = ", dynesty_logl)

    all(isfinite, [dark_siren_logl, bright_siren_logl, skymap_em_logl, stochastic_logl, joint_logl,
        dynesty_logl]) || error("science workflow example produced a non-finite value")
end

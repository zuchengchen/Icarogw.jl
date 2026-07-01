using BenchmarkTools
using HDF5
using Icarogw
using Random

rng = MersenneTwister(7)
model = SimplePowerLawPopulation()
data = simulate_population_data(rng, model; nevents=2, nsamples=128, ndetected=128, ntotal=1000, zmax=0.8)
schema = parameter_schema(SimplePowerLawPopulation)
theta = pack(schema, (
    alpha=2.0, beta=1.0, mmin=5.0, mmax=80.0,
    gamma=0.0, R0=25.0, H0=67.7, Om0=0.308,
))

println("likelihood evaluation")
display(@benchmark loglikelihood(SimplePowerLawPopulation, $data, $theta))

println("diagnostics and weighting")
display(@benchmark likelihood_diagnostics(SimplePowerLawPopulation, $data, $theta))

println("parallel posterior workspace")
display(@benchmark build_parallel_posterior($rng, $(data.posteriors), 64))

println("simulation mock generation")
display(@benchmark simulate_population_data($rng, $model; nevents=1, nsamples=64, ndetected=64, ntotal=500, zmax=0.5))

freqs = collect(range(20.0, 100.0; length=16))
omega_weights = precompute_omega_weights(MersenneTwister(9), freqs; tmp_min=5.0, tmp_max=80.0, n=256)

println("stochastic Omega_GW spectrum")
display(@benchmark spectral_siren_omega_gw($model, $omega_weights))

zgrid = collect(range(0.01, 0.3; length=64))
kc = KCorrection("bJ-glade+")
legacy_lf = LegacyGalaxyLuminosityFunction("K"; epsilon=0.8)
catalog_cosmology = FlatLambdaCDM(zmax=1.0)

println("catalog formula helpers")
display(@benchmark $kc($zgrid))
display(@benchmark background_effective_galaxy_density($legacy_lf, [-26.0, -22.0, -18.0]))
display(@benchmark em_likelihood_prior_differential_volume($zgrid, 0.1, 0.03, $catalog_cosmology; ptype="gaussian"))

catalog_runtime = mktempdir() do dir
    path = joinpath(dir, "toy_catalog.h5")
    runtime_nside = 1
    runtime_level = healpix_nside_to_level(runtime_nside)
    runtime_npix = 12 * runtime_nside^2
    runtime_uniq = [level_ipix_to_uniq(runtime_level, ipix) for ipix in 0:(runtime_npix - 1)]
    runtime_z = collect(range(0.05, 0.3; length=12))
    runtime_vals = [10.0 * iz + pix for iz in 1:length(runtime_z), pix in 1:runtime_npix]
    h5open(path, "w") do h
        group = create_group(h, "K")
        write(group, "mthr_moc_map", fill(20.0, runtime_npix))
        write(group, "uniq_moc_map", runtime_uniq)
        write(group, "z_grid", runtime_z)
        subgroup = create_group(group, "weighted")
        attrs(subgroup)["band"] = "K-glade+"
        attrs(subgroup)["epsilon"] = 0.8
        write(subgroup, "vals_interpolant", permutedims(runtime_vals))
        write(subgroup, "bg_vals_interpolant", permutedims(runtime_vals ./ 10))
    end
    IcarogwCatalog(path, "K", "weighted"; cosmology=catalog_cosmology)
end
catalog_ra = collect(range(0.0, 2pi; length=16))
catalog_dec = collect(range(-0.8, 0.8; length=16))
catalog_rows = get_NUNIQ_pixel(catalog_runtime, catalog_ra, catalog_dec)
catalog_z = collect(range(0.05, 0.3; length=16))
catalog_m1d = fill(36.0, length(catalog_z))
catalog_m2d = fill(24.0, length(catalog_z))
catalog_dl = luminosity_distance(catalog_cosmology, catalog_z)
catalog_rate = CBCCatalogVanillaRate(catalog_runtime, catalog_cosmology,
    ConditionalMassDistribution(PowerLaw(5.0, 80.0, -2.0), PowerLaw(5.0, 80.0, 1.0)),
    PowerLawRate(0.0))

println("catalog runtime")
display(@benchmark get_NUNIQ_pixel($catalog_runtime, $catalog_ra, $catalog_dec))
display(@benchmark effective_galaxy_number_interpolant($catalog_runtime, $catalog_z, $catalog_rows, $catalog_cosmology; average=true))
display(@benchmark event_logweights($catalog_rate, PosteriorSamples((mass_1=$catalog_m1d, mass_2=$catalog_m2d,
    luminosity_distance=$catalog_dl, sky_indices=$catalog_rows, prior=ones(length($catalog_z))))))

galaxy_dir = mktempdir()
galaxy_path = joinpath(galaxy_dir, "galaxy_catalog.h5")
galaxy_ra = collect(range(0.0, 2pi; length=24))
galaxy_dec = collect(range(-0.8, 0.8; length=24))
galaxy_zobs = collect(range(0.03, 0.28; length=24))
galaxy_mag = collect(range(12.0, 15.0; length=24))
galaxy_interp_z = collect(range(0.05, 0.3; length=12))
galaxy_npix = 12
create_hdf5(galaxy_path, (ra=galaxy_ra, dec=galaxy_dec, z=galaxy_zobs,
    sigmaz=fill(0.01, length(galaxy_ra)), m=galaxy_mag), "K", 1)
calculate_mthr!(galaxy_path; mthr_percentile=75)
h5open(galaxy_path, "r+") do h
    interp = create_group(h["catalog"], "dNgal_dzdOm_interpolant")
    attrs(interp)["epsilon"] = 0.8
    write(interp, "z_grid", galaxy_interp_z)
    for pix in 0:(galaxy_npix - 1)
        write(interp, "vals_pixel_$pix", log.([10.0 * iz + pix + 1 for iz in 1:length(galaxy_interp_z)]))
    end
end
galaxy_runtime = GalaxyCatalog(galaxy_path; cosmology=catalog_cosmology, epsilon=0.8)
galaxy_rows = galaxy_runtime.sky_indices[1:length(catalog_z)]

println("galaxy catalog runtime")
display(@benchmark GalaxyCatalog($galaxy_path; cosmology=$catalog_cosmology, epsilon=0.8))
display(@benchmark return_counts_map($galaxy_runtime))
display(@benchmark effective_galaxy_number_interpolant($galaxy_runtime, $catalog_z, $galaxy_rows, $catalog_cosmology; average=true))

pixel_dir = mktempdir()
pixel_data = (ra=collect(range(0.0, 2pi; length=24)),
    dec=collect(range(-0.8, 0.8; length=24)),
    z=collect(range(0.04, 0.28; length=24)),
    sigmaz=fill(0.01, 24),
    Kmag=collect(range(12.0, 15.0; length=24)))
pixel_zgrid = collect(range(1e-6, 0.35; length=8))

println("pixelated catalog preprocessing")
display(@benchmark begin
    dir = mktempdir()
    create_pixelated_catalogs(dir, 1, $pixel_data)
    filled = clear_empty_pixelated_files(dir, 1)
    for pixel in filled
        remove_nans_pixelated_files(dir, pixel, ["z", "sigmaz", "Kmag"], "K")
        calculate_mthr_pixelated_files(dir, pixel, "Kmag", "K", 1; mthr_percentile=75)
        get_redshift_grid_for_files(dir, pixel, "K", $catalog_cosmology; Nintegration=$pixel_zgrid, zcut=last($pixel_zgrid))
        calculate_interpolant_files(dir, $pixel_zgrid, pixel, "K", "weighted", "K-glade+", $catalog_cosmology, 0.8)
    end
    out = joinpath(dir, "catalog.h5")
    initialize_icarogw_catalog(dir, out, "K")
    build_icarogw_catalog_from_pixelated_files!(dir, out, "K", "weighted"; cosmology=$catalog_cosmology)
end)

ra = collect(range(0.0, 2pi; length=64))
dec = collect(range(-1.0, 1.0; length=64))
skymap_nside = 4
skymap_level = healpix_nside_to_level(skymap_nside)
skymap_uniq = [level_ipix_to_uniq(skymap_level, ipix) for ipix in 0:(12 * skymap_nside^2 - 1)]
toy_skymap = LigoSkyMap(skymap_uniq, fill(inv(4pi), length(skymap_uniq)),
    fill(100.0, length(skymap_uniq)), fill(10.0, length(skymap_uniq)))
dl = fill(100.0, length(ra))

println("skymap core")
display(@benchmark radec2skymap($ra, $dec, $skymap_nside; nest=true))
display(@benchmark evaluate_3d_posterior_likelihood($toy_skymap, $dl, $ra, $dec))

em_z = collect(range(0.08, 0.18; length=16))
em_dl = luminosity_distance(catalog_cosmology, em_z)
em_mass = ConditionalMassDistribution(PowerLaw(5.0, 80.0, -2.0), PowerLaw(5.0, 80.0, 1.0))
em_rate = CBCVanillaEMCounterpartRate(catalog_cosmology, em_mass, PowerLawRate(0.0); R0=2.0)
em_ps = PosteriorSamples((mass_1=fill(36.0, length(em_z)), mass_2=fill(24.0, length(em_z)),
    luminosity_distance=em_dl, z_EM=em_z .+ 0.002, prior=ones(length(em_z))))
em_inj = InjectionSet((mass_1=fill(36.0, length(em_z)), mass_2=fill(24.0, length(em_z)),
    luminosity_distance=em_dl, prior=ones(length(em_z))); ntotal=64, Tobs=1.0)
sky_em_rate = CBCLowLatencySkyMapEMCounterpartRate(catalog_cosmology, PowerLawRate(0.0), [toy_skymap]; R0=2.0)
sky_em_ps = PosteriorSamples((z_EM=em_z, right_ascension=fill(0.2, length(em_z)),
    declination=fill(0.1, length(em_z)), prior=ones(length(em_z))); event_name=:event1)
sky_em_inj = InjectionSet((luminosity_distance=em_dl, prior=ones(length(em_z))); ntotal=64, Tobs=1.0)

println("EM counterpart rates")
display(@benchmark event_logweights($em_rate, $em_ps))
display(@benchmark injection_logweights($em_rate, $em_inj))
display(@benchmark event_logweights($sky_em_rate, $sky_em_ps))
display(@benchmark injection_logweights($sky_em_rate, $sky_em_inj))

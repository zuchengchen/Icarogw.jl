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

println("catalog runtime")
display(@benchmark get_NUNIQ_pixel($catalog_runtime, $catalog_ra, $catalog_dec))
display(@benchmark effective_galaxy_number_interpolant($catalog_runtime, $catalog_z, $catalog_rows, $catalog_cosmology; average=true))

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

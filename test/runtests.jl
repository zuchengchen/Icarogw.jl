using Icarogw
using CSV
using DataFrames
using FITSIO
using HDF5
using Random
using Statistics: mean
using Test

_truthy(x) = x isa Bool ? x : lowercase(String(x)) == "true"

struct _ToyCatalogCosmology <: Icarogw.Cosmology.AbstractCosmology
    zmax::Float64
end
Icarogw.Cosmology.dvc_dz_dOmega(::_ToyCatalogCosmology, z::Real) = z^2 + 0.1
Icarogw.Cosmology.comoving_volume(::_ToyCatalogCosmology, z::Real) = 4pi * (z^3 / 3 + 0.1z)

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
    glf = GalaxyLuminosityFunction("K-glade+"; cosmology=c)
    @test galaxy_MF === GalaxyLuminosityFunction
    @test log_powerlaw_absM_rate === LogPowerLawAbsMagnitudeRate
    @test basic_absM_rate === AbstractAbsMagnitudeRate
    @test glf.Mstarobs ≈ -23.39 + 5log10(c.H0 / 100)
    @test luminosity_function_norm(glf, 0.2) > 0
    @test isfinite(log_luminosity_pdf(glf, -24.0, 0.2))
    @test length(sample_luminosity_function(MersenneTwister(4), glf, 4, 0.2)) == 4
    @test cred_interval(1.0) ≈ 0.6826894921370859 rtol=1e-12
    @test chirp_mass(30.0, 20.0) ≈ (30 * 20)^(3 / 5) / 50^(1 / 5)
    @test mass_ratio(30.0, 20.0) ≈ 2 / 3
    @test M2L(L2M(3.0128e28)) ≈ 3.0128e28 rtol=1e-6
    @test chi_effective_prior_from_aligned_spins(0.8, 1.0, 0.0) > 0
    @test chi_effective_prior_from_isotropic_spins(0.8, 1.0, 0.0) > 0
    @test chi_p_prior_from_isotropic_spins(0.8, 1.0, 0.3) > 0
end

@testset "skymap and HEALPix helpers" begin
    @test healpix_level_to_nside(0) == 1
    @test healpix_level_to_nside(3) == 8
    @test healpix_nside_to_level(8) == 3
    @test_throws ArgumentError healpix_nside_to_level(3)
    @test uniq_to_level_ipix(level_ipix_to_uniq(2, 7)) == (2, 7)

    ra = [0.1, 1.2, 2.4, 5.8]
    dec = [0.4, -0.2, 0.8, -1.0]
    pix = radec2indeces(ra, dec, 4; nest=true)
    pix0 = radec2indeces(ra, dec, 4; nest=true, zero_based=true)
    @test pix0 == pix .- 1
    ra_center, dec_center = indices2radec(pix, 4; nest=true)
    @test length(ra_center) == length(ra)
    @test all(0 .<= ra_center .< 2pi)
    @test all(-pi / 2 .<= dec_center .<= pi / 2)

    counts_map, area = radec2skymap(ra, dec, 4; nest=true)
    @test length(counts_map) == 12 * 4^2
    @test area ≈ 4pi / length(counts_map)
    @test sum(counts_map) * area ≈ 1.0

    level = 1
    nside = healpix_level_to_nside(level)
    uniq = [level_ipix_to_uniq(level, ipix) for ipix in 0:(12 * nside^2 - 1)]
    moc = MOCMap(collect(1:length(uniq)), uniq)
    rows = get_NUNIQ_pixel(moc, ra, dec)
    @test rows == radec2indeces(ra, dec, nside; nest=true)
    @test get_NUNIQ_pixel(moc, first(ra), first(dec)) == first(rows)

    probdensity = fill(inv(4pi), length(uniq))
    distmu = fill(100.0, length(uniq))
    distsigma = fill(10.0, length(uniq))
    sky = LigoSkyMap(uniq, probdensity, distmu, distsigma)
    intersect_em_pe(sky, ra, dec)
    @test sky.intersected
    @test sky.matched_rows == rows
    dl = fill(100.0, length(ra))
    ppost = evaluate_3d_posterior_intersected(sky, dl)
    expected_pdl = inv((2pi * 10.0^2)^2)
    @test ppost ≈ fill(inv(4pi) * expected_pdl, length(ra))
    skymap_area = pixel_area(nside)
    @test evaluate_3d_likelihood_intersected(sky, dl) ≈ ppost .* skymap_area ./ (dl .^ 2)
    ppost2, plike2 = evaluate_3d_posterior_likelihood(sky, dl, ra, dec)
    @test ppost2 ≈ ppost
    @test plike2 ≈ evaluate_3d_likelihood_intersected(sky, dl)
    draws = sample_3d_space(MersenneTwister(7), sky, 6)
    @test length.(draws) == (6, 6, 6)
    @test all(isfinite, draws[1])

    mktempdir() do dir
        path = joinpath(dir, "toy_skymap.fits")
        FITS(path, "w") do f
            FITSIO.write(f, ["UNIQ", "PROBDENSITY", "DISTMU", "DISTSIGMA"],
                [uniq, probdensity, distmu, distsigma])
        end
        from_fits = ligo_skymap(path)
        @test from_fits.uniq == uniq
        @test from_fits.probdensity == probdensity
        @test evaluate_3d_posterior_likelihood(from_fits, dl, ra, dec)[1] ≈ ppost
    end
end

@testset "catalog runtime readers" begin
    catalog_cosmology = FlatLambdaCDM(zmax=1.0)
    level = 0
    nside = healpix_level_to_nside(level)
    npix = 12 * nside^2
    uniq = [level_ipix_to_uniq(level, ipix) for ipix in 0:(npix - 1)]
    mthr = collect(range(18.0, 29.0; length=npix))
    z_grid = [0.1, 0.2, 0.4]
    dngal = [100.0 * iz + pix for iz in 1:length(z_grid), pix in 1:npix]
    bg_vals = [10.0 * iz + 0.5pix for iz in 1:length(z_grid), pix in 1:npix]

    mktempdir() do dir
        path = joinpath(dir, "icarogw_catalog.h5")
        h5open(path, "w") do h
            group = create_group(h, "K")
            write(group, "mthr_moc_map", mthr)
            write(group, "uniq_moc_map", uniq)
            write(group, "z_grid", z_grid)
            subgroup = create_group(group, "weighted")
            attrs(subgroup)["band"] = "K-glade+"
            attrs(subgroup)["epsilon"] = 0.8
            write(subgroup, "vals_interpolant", permutedims(dngal))
            write(subgroup, "bg_vals_interpolant", permutedims(bg_vals))
        end

        catalog = IcarogwCatalog(path, "K", "weighted"; cosmology=catalog_cosmology)
        @test icarogw_catalog === IcarogwCatalog
        @test catalog.band == "K-glade+"
        @test catalog.epsilon == 0.8
        @test catalog.dNgal_dzdOm_vals == dngal
        @test catalog.dNgal_dzdOm_vals_av ≈ vec(sum(dngal; dims=2)) ./ npix
        @test catalog.bg_vals_av ≈ vec(sum(bg_vals; dims=2)) ./ npix

        ra, dec = 0.2, 0.1
        row = get_NUNIQ_pixel(catalog, ra, dec)
        @test row == radec2indeces(ra, dec, nside; nest=true)
        @test get_NUNIQ_pixel(catalog, [ra], [dec]) == [row]
        @test calc_mthr(catalog, 0.2, row, catalog_cosmology; dl=100.0) ≈
              absolute_magnitude(mthr[row], 100.0, 0.0)
        @test calc_Mthr(catalog, [0.1, 0.2], row, catalog_cosmology; dl=fill(100.0, 2)) ≈
              absolute_magnitude.(fill(mthr[row], 2), fill(100.0, 2), 0.0)

        gc, bg = effective_galaxy_number_interpolant(catalog, 0.2, row, catalog_cosmology; dl=100.0)
        expected_mthr = absolute_magnitude(mthr[row], 100.0, 0.0)
        @test gc ≈ dngal[2, row]
        @test bg ≈ background_effective_galaxy_density(catalog.luminosity_function, expected_mthr,
            0.2, catalog.abs_magnitude_rate) * dvc_dz_dOmega(catalog_cosmology, 0.2)

        gc_interp, _ = effective_galaxy_number_interpolant(catalog, 0.15, row, catalog_cosmology; dl=100.0)
        @test gc_interp ≈ (dngal[1, row] + dngal[2, row]) / 2

        gc_av, bg_av = effective_galaxy_number_interpolant(catalog, 0.2, row, catalog_cosmology; average=true)
        @test gc_av ≈ catalog.dNgal_dzdOm_vals_av[2]
        @test bg_av ≈ catalog.bg_vals_av[2]

        gc_out, bg_out = effective_galaxy_number_interpolant(catalog, 0.5, row, catalog_cosmology; dl=100.0)
        @test gc_out == 0.0
        @test bg_out ≈ background_effective_galaxy_density(catalog.luminosity_function, -Inf,
            0.5, catalog.abs_magnitude_rate) * dvc_dz_dOmega(catalog_cosmology, 0.5)

        make_me_empty!(catalog, catalog_cosmology)
        @test all(iszero, catalog.dNgal_dzdOm_vals)
        @test all(iszero, catalog.dNgal_dzdOm_vals_av)
        @test catalog.bg_vals_av ≈ background_effective_galaxy_density(catalog.luminosity_function,
            -Inf, z_grid, catalog.abs_magnitude_rate) .* dvc_dz_dOmega(catalog_cosmology, z_grid)
    end

    mktempdir() do dir
        path = joinpath(dir, "gwcosmo_catalog.h5")
        offset = 0.25
        pz_empty = [1.0, 2.0, 3.0]
        combined = [10.0, 20.0, 40.0]
        vals = [100.0 * pix + iz for iz in 1:length(z_grid), pix in 1:npix]
        h5open(path, "w") do h
            attrs(h)["opts"] = "{'offset': $offset}"
            write(h, "z_array", z_grid)
            write(h, "empty_catalogue", log.(pz_empty .+ offset))
            write(h, "combined_pixels", log.(combined .+ offset))
            for pix in 0:(npix - 1)
                write(h, string(pix), log.(vals[:, pix + 1] .+ offset))
            end
        end

        catalog = GwcosmoCatalog(path, nside, "K-glade+", 0.8; cosmology=catalog_cosmology)
        @test gwcosmo_catalog === GwcosmoCatalog
        @test catalog.z_grid == z_grid
        @test catalog.pz_empty ≈ pz_empty
        @test catalog.dNgal_dzdOm_vals_av ≈ combined
        @test catalog.dNgal_dzdOm_vals ≈ vals

        ra, dec = 0.2, 0.1
        row = get_NUNIQ_pixel(catalog, ra, dec)
        @test row == radec2indeces(ra, dec, nside; nest=true)
        gc, bg = effective_galaxy_number_interpolant(catalog, 0.2, row, catalog_cosmology)
        @test gc ≈ vals[2, row]
        @test bg == 0.0

        gc_av, bg_av = effective_galaxy_number_interpolant(catalog, [0.1, 0.2], row, catalog_cosmology; average=true)
        @test gc_av ≈ combined[1:2]
        @test bg_av == zeros(2)

        make_me_empty!(catalog)
        @test catalog.dNgal_dzdOm_vals_av ≈ pz_empty
        @test all(catalog.dNgal_dzdOm_vals[:, j] ≈ pz_empty for j in axes(catalog.dNgal_dzdOm_vals, 2))
    end

    mktempdir() do dir
        path = joinpath(dir, "galaxy_catalog.h5")
        ra = [0.2, 1.0, 1.1, Inf]
        dec = [0.1, -0.1, -0.12, 0.0]
        z = [0.1, 0.2, 0.22, 0.3]
        sigmaz = [0.01, 0.02, 0.03, 0.04]
        mag = [12.0, 13.0, 14.0, 15.0]
        create_hdf5(path, (ra=ra, dec=dec, z=z, sigmaz=sigmaz, m=mag), "K", 1)
        catalog = GalaxyCatalog(path; epsilon=0.8)
        @test galaxy_catalog === GalaxyCatalog
        @test load_hdf5 === GalaxyCatalog
        @test catalog.band == "K"
        @test catalog.npixels == 12
        @test length(catalog.ra) == 3
        @test catalog.sky_indices == radec2indeces(ra[1:3], dec[1:3], 1)
        @test sum(return_counts_map(catalog)) == 3

        catalog_mthr = calculate_mthr!(path; mthr_percentile=50)
        @test catalog_mthr.mthr_map !== nothing
        @test length(catalog_mthr.mthr_map) == catalog_mthr.npixels
        @test length(catalog_mthr.m) == 2
        @test calc_mthr(catalog_mthr, 0.2, catalog_mthr.sky_indices[1], catalog_cosmology; dl=100.0) ≈
              absolute_magnitude(catalog_mthr.mthr_map[catalog_mthr.sky_indices[1]], 100.0,
                  DeprecatedKCorrection("K")(0.2))

        h5open(path, "r+") do h
            group = h["catalog"]
            interp = create_group(group, "dNgal_dzdOm_interpolant")
            attrs(interp)["epsilon"] = 0.8
            write(interp, "z_grid", z_grid)
            for pix in 0:(catalog_mthr.npixels - 1)
                values = [10.0 * iz + pix + 1 for iz in 1:length(z_grid)]
                write(interp, "vals_pixel_$pix", log.(values))
            end
        end
        catalog_loaded = GalaxyCatalog(path; cosmology=catalog_cosmology)
        row = catalog_loaded.sky_indices[1]
        gc, bg = effective_galaxy_number_interpolant(catalog_loaded, 0.2, row, catalog_cosmology; dl=100.0)
        @test gc ≈ 20.0 + row
        @test bg >= 0.0
        gc_av, _ = effective_galaxy_number_interpolant(catalog_loaded, 0.2, row, catalog_cosmology; average=true, dl=100.0)
        @test gc_av ≈ mean([20.0 + pix for pix in 1:catalog_loaded.npixels])

        empty_path = joinpath(dir, "galaxy_empty.h5")
        create_hdf5(empty_path, (ra=ra[1:3], dec=dec[1:3], z=z[1:3], sigmaz=sigmaz[1:3], m=mag[1:3]), "K", 1)
        empty_catalog = calculate_mthr!(empty_path; mthr_percentile="empty")
        empty_loaded = GalaxyCatalog(empty_path; cosmology=catalog_cosmology, epsilon=0.8)
        @test empty_catalog.mthr_empty
        @test empty_loaded.mthr_empty
        gc_empty, bg_empty = effective_galaxy_number_interpolant(empty_loaded, 0.2, 1, catalog_cosmology)
        @test gc_empty == 0.0
        @test bg_empty > 0.0
    end
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
    catalog_ref = CSV.File(joinpath(refdir, "reference_catalog_luminosity.csv")) |> DataFrame
    catalog_lf = GalaxyLuminosityFunction(first(catalog_ref.band); cosmology=FlatLambdaCDM(H0=100 * first(catalog_ref.little_h)))
    abs_rate = LogPowerLawAbsMagnitudeRate(first(catalog_ref.epsilon))
    @test catalog_lf.Mminobs ≈ first(catalog_ref.Mminobs) rtol=2e-14
    @test catalog_lf.Mmaxobs ≈ first(catalog_ref.Mmaxobs) rtol=2e-14
    @test catalog_lf.Mstarobs ≈ first(catalog_ref.Mstarobs) rtol=2e-14
    @test catalog_lf.phistarobs ≈ first(catalog_ref.phistarobs) rtol=2e-14
    @test first.(evolved_luminosity_parameters(catalog_lf, catalog_ref.redshift)) ≈ catalog_ref.phistar_z rtol=2e-14
    @test last.(evolved_luminosity_parameters(catalog_lf, catalog_ref.redshift)) ≈ catalog_ref.Mstar_z rtol=2e-14
    @test log_luminosity_function(catalog_lf, catalog_ref.M_abs, catalog_ref.redshift) ≈ catalog_ref.log_luminosity_function rtol=2e-14
    @test luminosity_function(catalog_lf, catalog_ref.M_abs, catalog_ref.redshift) ≈ catalog_ref.luminosity_function rtol=2e-14
    @test log_abs_magnitude_rate(abs_rate, catalog_lf, catalog_ref.M_abs) ≈ catalog_ref.log_abs_magnitude_rate rtol=2e-14
    @test abs_magnitude_rate(abs_rate, catalog_lf, catalog_ref.M_abs) ≈ catalog_ref.abs_magnitude_rate rtol=2e-14
    @test background_effective_galaxy_density(catalog_lf, catalog_ref.Mthr, catalog_ref.redshift, abs_rate) ≈
          catalog_ref.background_effective_density rtol=4e-4 atol=1e-12
    @test background_effective_galaxy_density(catalog_lf, first(catalog_ref.Mthr), first(catalog_ref.redshift);
        epsilon=first(catalog_ref.epsilon)) ≈ first(catalog_ref.background_effective_density) rtol=4e-4

    kcorr_ref = CSV.File(joinpath(refdir, "reference_catalog_kcorr.csv")) |> DataFrame
    for row in eachrow(kcorr_ref)
        if row.family == "modern"
            correction = KCorrection(row.band)
            if endswith(row.band, "upglade")
                @test correction(row.z; k0=0.1 + 0.05 * (row.i - 1),
                    dkbydz=[1.0, -0.5, 0.25, 0.75][row.i], z0=0.05 * (row.i - 1)) ≈ row.kcorr rtol=2e-14
            else
                @test correction(row.z) ≈ row.kcorr rtol=2e-14
            end
        else
            @test DeprecatedKCorrection(row.band)(row.z) ≈ row.kcorr rtol=2e-14
        end
    end
    @test kcorr === KCorrection
    @test kcorr_dep === DeprecatedKCorrection
    @test_throws ArgumentError KCorrection("unknown-band")
    @test_throws ArgumentError KCorrection("W1-upglade")(0.1)

    em_ref = CSV.File(joinpath(refdir, "reference_catalog_em.csv")) |> DataFrame
    toy_catalog_cosmology = _ToyCatalogCosmology(1.0)
    for row in eachrow(em_ref)
        @test user_normal(row.z, row.zobs, row.sigmaz) ≈ row.normal_pdf rtol=2e-14
        @test em_likelihood_prior_differential_volume(row.z, row.zobs, row.sigmaz, toy_catalog_cosmology;
            Numsigma=row.Numsigma, ptype=row.ptype) ≈ row.value rtol=4e-10 atol=1e-12
    end
    for ptype in unique(em_ref.ptype)
        rows = em_ref[em_ref.ptype .== ptype, :]
        @test EM_likelihood_prior_differential_volume(rows.z, first(rows.zobs), first(rows.sigmaz),
            toy_catalog_cosmology; Numsigma=first(rows.Numsigma), ptype=ptype) ≈ rows.value rtol=4e-10 atol=1e-12
    end

    legacy_ref = CSV.File(joinpath(refdir, "reference_catalog_legacy_luminosity.csv")) |> DataFrame
    legacy_lf = LegacyGalaxyLuminosityFunction(first(legacy_ref.band);
        cosmology=FlatLambdaCDM(H0=100 * first(legacy_ref.little_h)),
        epsilon=first(legacy_ref.epsilon))
    @test galaxy_MF_dep === LegacyGalaxyLuminosityFunction
    @test legacy_lf.Mminobs ≈ first(legacy_ref.Mminobs) rtol=2e-14
    @test legacy_lf.Mmaxobs ≈ first(legacy_ref.Mmaxobs) rtol=2e-14
    @test legacy_lf.Mstarobs ≈ first(legacy_ref.Mstarobs) rtol=2e-14
    @test legacy_lf.phistarobs ≈ first(legacy_ref.phistarobs) rtol=2e-14
    @test legacy_lf.norm ≈ first(legacy_ref.norm) rtol=2e-10
    @test log_luminosity_function(legacy_lf, legacy_ref.M_abs) ≈ legacy_ref.log_luminosity_function rtol=2e-14
    @test luminosity_function(legacy_lf, legacy_ref.M_abs) ≈ legacy_ref.luminosity_function rtol=2e-14
    @test log_luminosity_pdf(legacy_lf, legacy_ref.M_abs) ≈ legacy_ref.log_luminosity_pdf rtol=2e-14
    @test luminosity_pdf(legacy_lf, legacy_ref.M_abs) ≈ legacy_ref.luminosity_pdf rtol=2e-14
    @test background_effective_galaxy_density(legacy_lf, legacy_ref.Mthr) ≈
          legacy_ref.background_effective_density rtol=2e-10 atol=1e-12
    @test length(sample_luminosity_function(MersenneTwister(5), legacy_lf, 3)) == 3
    @test_throws ArgumentError LegacyGalaxyLuminosityFunction("bad-band")
    @test_throws ArgumentError background_effective_galaxy_density(LegacyGalaxyLuminosityFunction("K"), -22.0)

    conv_ref = only(CSV.File(joinpath(refdir, "reference_conversions_core.csv")) |> DataFrame |> eachrow)
    @test chirp_mass(conv_ref.m1, conv_ref.m2) ≈ conv_ref.chirp_mass rtol=1e-14
    @test mass_ratio(conv_ref.m1, conv_ref.m2) ≈ conv_ref.mass_ratio rtol=1e-14
    @test f_gw_isco(conv_ref.m1, conv_ref.m2) ≈ conv_ref.f_gw_isco rtol=1e-14
    @test L2M(conv_ref.L) ≈ conv_ref.L2M rtol=1e-14 atol=1e-14
    @test M2L(conv_ref.M) ≈ conv_ref.M2L rtol=1e-14
    @test chi_eff_from_spins(conv_ref.chi1, conv_ref.chi2, conv_ref.cos1, conv_ref.cos2, conv_ref.q) ≈ conv_ref.chi_eff rtol=1e-14
    @test chi_p_from_spins(conv_ref.chi1, conv_ref.chi2, conv_ref.cos1, conv_ref.cos2, conv_ref.q) ≈ conv_ref.chi_p rtol=1e-14
    conditional_chi_p = chi_p_prior_given_chi_eff_q(MersenneTwister(321),
        conv_ref.q_spin_1, 1.0, conv_ref.xeff_spin_1, conv_ref.xp_spin_1; ndraws=2000)
    @test conditional_chi_p ≈ conv_ref.conditional_chi_p_prior rtol=0.25
    joint_spin = joint_prior_from_isotropic_spins(MersenneTwister(322),
        [conv_ref.q_spin_1, conv_ref.q_spin_2], 1.0,
        [conv_ref.xeff_spin_1, conv_ref.xeff_spin_2],
        [conv_ref.xp_spin_1, conv_ref.xp_spin_2]; ndraws=2000)
    @test joint_spin ≈ [conv_ref.joint_spin_prior_1, conv_ref.joint_spin_prior_2] rtol=0.25

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

    stochastic_ref = CSV.File(joinpath(refdir, "reference_stochastic_dedf.csv")) |> DataFrame
    freqs = stochastic_ref.freq
    values = dedf(first(stochastic_ref.Mtot), freqs; eta=first(stochastic_ref.eta), chi=first(stochastic_ref.chi))
    @test values ≈ stochastic_ref.dEdf rtol=2e-14
    @test dedf(first(stochastic_ref.Mtot), first(freqs); eta=first(stochastic_ref.eta), chi=first(stochastic_ref.chi)) ≈ first(stochastic_ref.dEdf) rtol=2e-14

    sim_ref = CSV.File(joinpath(refdir, "reference_simulation_utils.csv")) |> DataFrame
    @test chirp_mass_detector(sim_ref.m1, sim_ref.m2, sim_ref.z) ≈ sim_ref.chirp_mass_det rtol=1e-14
    @test f_gw(sim_ref.m1, sim_ref.m2, sim_ref.z) ≈ sim_ref.f_gw rtol=1e-14
    @test snr_samples_flat(sim_ref.z; alpha=2.0) ≈ sim_ref.snr_flat_alpha2 rtol=1e-14
    likelihood = likelihood_evaluation(
        [10.0, 14.0, 18.0],
        [0.60, 0.50, 0.80],
        [24.0, 30.0, 42.0],
        [0.5, 0.8, 1.1],
        [11.0, 13.0, 19.0],
        [0.62, 0.48, 0.78],
        [24.1, 29.8, 42.5],
        [0.55, 0.75, 1.05],
    )
    @test likelihood ≈ sim_ref.likelihood rtol=1e-12
    @test check_bounds_1d([-1.0, 0.5, 2.0], 0.0, 1.0) == _truthy.(sim_ref.bounds_1d)
    @test check_bounds_2d([1.0, 0.5, 2.0], [0.5, 0.7, 1.0], [0.1, NaN, 0.3]) == _truthy.(sim_ref.bounds_2d)
    @test snr_and_freq_cut(sim_ref.m1, sim_ref.m2, sim_ref.z, [13.0, 9.0, 20.0]; snr_threshold=12.0, fgw_cut=15.0) ==
          findall(_truthy.(sim_ref.passes_snr_freq))
    @test snr_cut_flat([13.0, 9.0, 20.0]; snr_threshold=12.0) == findall(_truthy.(sim_ref.passes_snr_flat))

    priors_rates_ref = CSV.File(joinpath(refdir, "reference_priors_rates.csv")) |> DataFrame
    @test highpass_filter(priors_rates_ref.x, 5.0, 2.0) ≈ priors_rates_ref.highpass rtol=1e-14
    @test lowpass_filter(priors_rates_ref.x, 8.0, 2.0) ≈ priors_rates_ref.lowpass rtol=1e-14
    @test notch_filter(priors_rates_ref.x, 5.0, 2.0, 8.0, 2.0, 0.4) ≈ priors_rates_ref.notch rtol=1e-14
    @test mixed_linear_function(priors_rates_ref.x ./ 10.0, 0.2, 0.8) ≈ priors_rates_ref.mixed_linear rtol=1e-14
    @test mixed_double_sigmoid_function(priors_rates_ref.x ./ 10.0, 0.55, 0.15, 0.2, 0.8) ≈
          priors_rates_ref.mixed_sigmoid rtol=1e-14
    lowpass_evolving = LowpassSmoothedProbEvolving(PowerLaw(5.0, 12.0, -2.0), 2.0)
    @test logpdf(lowpass_evolving, priors_rates_ref.x) ≈ priors_rates_ref.lowpass_evolving_logpdf rtol=2e-12
    @test 0 <= cdf(lowpass_evolving, 7.0) <= 1
    abs_lum = AbsLuminosityPowerLawInMagnitude(-23.0, -16.0, -1.1)
    @test logpdf(abs_lum, priors_rates_ref.M_abs) ≈ priors_rates_ref.abs_l_powerlaw_logpdf rtol=2e-14
    @test cdf(abs_lum, priors_rates_ref.M_abs) ≈ priors_rates_ref.abs_l_powerlaw_cdf rtol=2e-14
    pl_pl = MixtureMassPrior((PowerLawStationary(1.5, 5.0, 50.0), PowerLawStationary(2.5, 8.0, 80.0)), [0.65, 0.35])
    @test logpdf(pl_pl, priors_rates_ref.mix_mass) ≈ priors_rates_ref.pl_pl_logpdf rtol=2e-14
    pl_pl_g = MixtureMassPrior((PowerLawStationary(1.5, 5.0, 50.0), PowerLawStationary(2.5, 8.0, 80.0),
        GaussianStationary(30.0, 4.0, 5.0)), [0.45, 0.35, 0.20])
    @test logpdf(pl_pl_g, priors_rates_ref.mix_mass) ≈ priors_rates_ref.pl_pl_g_logpdf rtol=2e-14
    pl_g_z = RedshiftMixtureMassPrior((PowerLawLinear(1.5, 0.2, 5.0, 0.5, 60.0, 1.0),
        GaussianLinear(25.0, 2.0, 4.0, 0.5, 5.0)),
        z -> (w = mixed_linear_function(z, 0.75, 0.35); [w, 1 - w]))
    @test logpdf(pl_g_z, priors_rates_ref.mix_mass, priors_rates_ref.mix_redshift) ≈ priors_rates_ref.pl_g_z_logpdf rtol=2e-14
    g_g_z = RedshiftMixtureMassPrior((GaussianLinear(20.0, 1.0, 3.0, 0.4, 5.0),
        GaussianLinear(40.0, -2.0, 5.0, 0.2, 5.0)),
        z -> (w = mixed_linear_function(z, 0.6, 0.3); [w, 1 - w]))
    @test logpdf(g_g_z, priors_rates_ref.mix_mass, priors_rates_ref.mix_redshift) ≈ priors_rates_ref.g_g_z_logpdf rtol=2e-14
    pair_base = PowerLaw(5.0, 60.0, -2.0)
    paired_dip = paired_massratio_dip(pair_base; beta=1.2, bottomsmooth=2.0, topsmooth=5.0,
        leftdip=10.0, rightdip=20.0, leftdipsmooth=2.0, rightdipsmooth=3.0, deep=0.4)
    @test logpdf(paired_dip, priors_rates_ref.m1_pair, priors_rates_ref.m2_pair) ≈
          priors_rates_ref.paired_dip_logpdf atol=0.12
    paired_general = paired_massratio_dip_general(pair_base; beta_bottom=0.5, beta_top=2.0,
        bottomsmooth=2.0, topsmooth=5.0, leftdip=10.0, rightdip=20.0, leftdipsmooth=2.0, rightdipsmooth=3.0, deep=0.4)
    @test logpdf(paired_general, priors_rates_ref.m1_pair, priors_rates_ref.m2_pair) ≈
          priors_rates_ref.paired_dip_general_logpdf atol=0.12
    paired_farah = paired_massratio_bpl_dip_farah_2022(alpha_1=1.5, alpha_2=3.0, mmin=5.0, mmax=60.0,
        beta_bottom=0.5, beta_top=2.0, bottomsmooth=2.0, topsmooth=5.0, leftdip=12.0, rightdip=24.0,
        leftdipsmooth=2.0, rightdipsmooth=3.0, deep=0.4)
    @test logpdf(paired_farah, priors_rates_ref.m1_pair, priors_rates_ref.m2_pair) ≈
          priors_rates_ref.paired_farah_logpdf atol=0.12
    paired_bplmulti = paired_massratio_bplmulti_dip(alpha_1=1.5, alpha_2=3.0, mmin=5.0, mmax=60.0,
        beta_bottom=0.5, beta_top=2.0, bottomsmooth=2.0, topsmooth=5.0, leftdip=12.0, rightdip=24.0,
        leftdipsmooth=2.0, rightdipsmooth=3.0, deep=0.4, mu_g_low=10.0, sigma_g_low=1.5,
        lambda_g_low=0.4, mu_g_high=35.0, sigma_g_high=3.0, lambda_g=0.15)
    @test logpdf(paired_bplmulti, priors_rates_ref.m1_pair, priors_rates_ref.m2_pair) ≈
          priors_rates_ref.paired_bplmulti_logpdf atol=0.12
    paired_triple = paired_bpl_triplepeak_dip(alpha_1=1.5, alpha_2=3.0, mmin=5.0, mmax=60.0,
        beta_bottom=0.5, beta_top=2.0, bottomsmooth=2.0, topsmooth=5.0, leftdip=12.0, rightdip=24.0,
        leftdipsmooth=2.0, rightdipsmooth=3.0, deep=0.4, mu_g_1=9.0, sigma_g_1=1.0,
        lambda_g=0.2, mu_g_2=25.0, sigma_g_2=2.0, lambda_1=0.3,
        mu_g_3=40.0, sigma_g_3=3.0, lambda_2=1.0)
    @test logpdf(paired_triple, priors_rates_ref.m1_pair, priors_rates_ref.m2_pair) ≈
          priors_rates_ref.paired_triple_logpdf atol=0.12
    paired_bplmulti_conditioned = paired_massratio_bplmulti_dip_conditioned(alpha_1=1.5, alpha_2=3.0,
        mmin=5.0, mmax=60.0, beta_bottom=0.5, beta_top=2.0, bottomsmooth=2.0, topsmooth=5.0,
        leftdip=12.0, rightdip=24.0, leftdipsmooth=2.0, rightdipsmooth=3.0, deep=0.4,
        mu_g_low=10.0, sigma_g_low=1.5, lambda_g_low=0.4, mu_g_high=35.0, sigma_g_high=3.0, lambda_g=0.15)
    @test logpdf(paired_bplmulti_conditioned, priors_rates_ref.m1_pair, priors_rates_ref.m2_pair) ≈
          priors_rates_ref.paired_bplmulti_conditioned_logpdf rtol=0.02
    bin_model = bin_model_2d(5.0, 45.0, [1.0, 2.0, 3.0])
    @test logpdf(bin_model, priors_rates_ref.m1_pair, priors_rates_ref.m2_pair) ≈ priors_rates_ref.bin_model_logpdf rtol=2e-14
    spin_gaussian = GaussianComponentSpinPrior(0.25, 0.35, 0.2, 0.25, 0.5, 0.4)
    @test logpdf(spin_gaussian, priors_rates_ref.chi1, priors_rates_ref.chi2,
        priors_rates_ref.cos1, priors_rates_ref.cos2) ≈ priors_rates_ref.spin_gaussian_logpdf rtol=2e-14
    spin_evolving = EvolvingGaussianSpinPrior(0.15, 0.2, 0.002, 0.001, 0.6, 0.3)
    @test logpdf(spin_evolving, priors_rates_ref.chi1, priors_rates_ref.chi2, priors_rates_ref.cos1,
        priors_rates_ref.cos2, priors_rates_ref.mass1_source, priors_rates_ref.mass2_source) ≈
          priors_rates_ref.spin_evolving_logpdf rtol=2e-14
    spin_beta_gaussian = BetaWindowGaussianSpinPrior(0.8, 0.2, 40.0, 2.0, 3.0, 0.35, 0.2, 0.5, 0.4)
    @test logpdf(spin_beta_gaussian, priors_rates_ref.chi1, priors_rates_ref.chi2, priors_rates_ref.cos1,
        priors_rates_ref.cos2, priors_rates_ref.mass1_source, priors_rates_ref.mass2_source) ≈
          priors_rates_ref.spin_beta_gaussian_logpdf rtol=2e-14
    spin_beta_beta = BetaWindowBetaSpinPrior(0.8, 0.2, 40.0, 2.0, 3.0, 4.0, 2.5, 0.5, 0.4)
    @test logpdf(spin_beta_beta, priors_rates_ref.chi1, priors_rates_ref.chi2, priors_rates_ref.cos1,
        priors_rates_ref.cos2, priors_rates_ref.mass1_source, priors_rates_ref.mass2_source) ≈
          priors_rates_ref.spin_beta_beta_logpdf rtol=2e-14
    pseob = PSEOBGaussianPrior(0.1, 1.2, -0.2, 1.5, 0.25)
    @test logpdf(pseob, priors_rates_ref.domega220, priors_rates_ref.dtau220) ≈ priors_rates_ref.pseob_logpdf rtol=2e-14
    eco = ECOTotallyReflectiveSpinPrior(2.0, 3.0, 1e-10, 0.35, 0.03)
    @test logpdf(eco, priors_rates_ref.chi1, priors_rates_ref.chi2) ≈ priors_rates_ref.eco_logpdf rtol=2e-14
    bivar = Bivariate2DGaussian(
        x1min=-1.0,
        x1max=1.0,
        x1mean=0.0,
        x2min=-2.0,
        x2max=2.0,
        x2mean=0.1,
        x1variance=0.5,
        x12covariance=0.1,
        x2variance=1.0,
    )
    @test logpdf(bivar, priors_rates_ref.x1, priors_rates_ref.x2) ≈ priors_rates_ref.bivariate_logpdf rtol=2e-14
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
    @test absL_PL_inM === AbsLuminosityPowerLawInMagnitude
    @test_throws ArgumentError BetaWindowBetaSpinPrior(0.8, 0.2, 40.0, 1.0, 3.0, 4.0, 2.5, 0.5, 0.4)
    @test_throws ArgumentError paired_massratio_bpl_dip_farah_2022(alpha_1=1.5, alpha_2=3.0, mmin=5.0, mmax=60.0,
        beta_bottom=0.5, beta_top=2.0, bottomsmooth=2.0, topsmooth=5.0, leftdip=4.0, rightdip=24.0,
        leftdipsmooth=2.0, rightdipsmooth=3.0, deep=0.4)

    cm = ConditionalMassDistribution(PowerLaw(5, 80, -2), PowerLaw(5, 80, 1))
    @test isfinite(logpdf(cm, 30.0, 20.0))
    @test logpdf(cm, 20.0, 30.0) == -Inf
    plz = PowerLawLinear(2.0, 0.2, 5.0, 0.5, 80.0, 1.0)
    @test isfinite(logpdf(plz, 30.0, 0.4))
    @test isfinite(logpdf(LowpassSmoothedProbEvolving(plz, 2.0), 30.0, 0.4))
    gz = GaussianLinear(30.0, 2.0, 5.0, 0.1, 5.0)
    @test isfinite(logpdf(gz, 31.0, 0.5))
    mixz = MixtureMassPrior((PowerLawStationary(2.0, 5.0, 80.0), gz), [0.7, 0.3])
    @test isfinite(logpdf(mixz, 30.0, 0.2))
    rcm = RedshiftConditionalMassDistribution(plz, PowerLaw(5.0, 80.0, 1.0))
    @test isfinite(logpdf(rcm, 30.0, 20.0, 0.4))
    @test logpdf(rcm, 20.0, 30.0, 0.4) == -Inf

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
    @test expected_number_detections(model, tiny.injections) ≈ diag_tiny.N_expected
    @test injection_logweights(model, tiny.injections) isa Vector{Float64}
    @test event_logweights(model, first(tiny.posteriors.events)) isa Vector{Float64}
    @test effective_sample_size(model, tiny.injections) ≈ diag_tiny.injection_neff
    @test effective_sample_size(model, first(tiny.posteriors.events)) ≈ first(diag_tiny.per_event_neff)
    inj_subset = subset_injections(tiny.injections, [true, false, true, false])
    @test length(inj_subset.prior) == 2
    @test inj_subset.ntotal == tiny.injections.ntotal
    ps_subset = subset_posterior_samples(first(tiny.posteriors.events), [1, 3])
    @test length(ps_subset.prior) == 2
    @test ps_subset.event_name == first(tiny.posteriors.events).event_name
    counterpart = add_counterpart(first(tiny.posteriors.events), [0.11, 0.12, 0.13])
    @test :z_EM in counterpart.names
    @test collect(column(counterpart, :z_EM)) == [0.11, 0.12, 0.13]
    @test first(tiny.posteriors.events).names == [:mass_1, :mass_2, :luminosity_distance]
    ra = [0.1, 1.1, 2.2]
    dec = [0.2, -0.3, 0.4]
    sky_ps = PosteriorSamples((right_ascension=ra, declination=dec, prior=ones(3)); event_name=:sky)
    pixelized_ps = pixelize(sky_ps, 2; nest=true)
    @test :sky_indices in pixelized_ps.names
    @test collect(column(pixelized_ps, :sky_indices)) == radec2indeces(ra, dec, 2; nest=true)
    @test collect(column(pixelize(sky_ps, 2; nest=true, zero_based=true), :sky_indices)) ==
          radec2indeces(ra, dec, 2; nest=true, zero_based=true)
    @test :sky_indices ∉ sky_ps.names
    pixelized_set = pixelize(PosteriorSampleSet(sky_ps, sky_ps), 2; nest=true)
    @test all(:sky_indices in event.names for event in pixelized_set.events)
    level = healpix_nside_to_level(2)
    uniq = [level_ipix_to_uniq(level, ipix) for ipix in 0:(12 * 2^2 - 1)]
    moc = MOCMap(collect(1:length(uniq)), uniq)
    catalog_pixelized_ps = pixelize_with_catalog(sky_ps, moc)
    @test collect(column(catalog_pixelized_ps, :sky_indices)) == get_NUNIQ_pixel(moc, ra, dec)
    catalog_pixelized_set = pixelize_with_catalog(PosteriorSampleSet(sky_ps, sky_ps), moc)
    @test all(column(event, :sky_indices) == column(catalog_pixelized_ps, :sky_indices) for event in catalog_pixelized_set.events)
    sky_inj = InjectionSet((right_ascension=ra, declination=dec, prior=ones(3)); ntotal=5, Tobs=1.5)
    pixelized_inj = pixelize(sky_inj, 2; nest=true)
    @test pixelized_inj.ntotal == sky_inj.ntotal
    @test pixelized_inj.Tobs == sky_inj.Tobs
    @test collect(column(pixelized_inj, :sky_indices)) == radec2indeces(ra, dec, 2; nest=true)
    @test collect(column(pixelize_with_catalog(sky_inj, moc), :sky_indices)) == get_NUNIQ_pixel(moc, ra, dec)
    @test_throws ArgumentError pixelize(ps_subset, 2)
    parallel = build_parallel_posterior(MersenneTwister(11), PosteriorSampleSet(ps_subset, first(tiny.posteriors.events)), 3)
    @test parallel isa ParallelPosterior
    @test parallel.event_names == [ps_subset.event_name, first(tiny.posteriors.events).event_name]
    @test parallel.names == ps_subset.names
    @test size(parallel.values[:mass_1]) == (2, 3)
    @test parallel.Ns_array == [2.0, 3.0]
    @test parallel.weights_mask[1, 3] == true
    @test parallel.weights_mask[2, 3] == false
    @test parallel.prior[1, 3] in ps_subset.prior
    @test_throws ArgumentError build_parallel_posterior(PosteriorSampleSet(ps_subset, counterpart), 2)
    @test_throws ArgumentError add_counterpart(ps_subset, [0.1])
    rw_inj = reweight_injections(MersenneTwister(7), model, tiny.injections, 3)
    @test length(rw_inj.prior) == 3
    @test rw_inj.ntotal == tiny.injections.ntotal
    rw_ps = reweight_posterior_samples(MersenneTwister(8), model, first(tiny.posteriors.events), 2)
    @test length(rw_ps.prior) == 2
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

    rng_noise = MersenneTwister(2027)
    Md_obs, q_obs, theta_obs = noise(rng_noise, [25.0, 30.0], [0.7, 0.8], [0.4, 0.5], [12.0, 15.0])
    @test length(Md_obs) == 2
    @test length(q_obs) == 2
    @test length(theta_obs) == 2
    @test isfinite(chirp_mass_noise(MersenneTwister(1), 25.0, 12.0))
    snr_det = snr_samples_detector(MersenneTwister(2), [33.0, 28.0], [22.0, 20.0], [900.0, 1100.0]; theta=[0.8, 1.0])
    snr_src = snr_samples_source(MersenneTwister(2), [30.0, 24.0], [20.0, 18.0], [0.1, 0.1]; theta=[0.8, 1.0])
    @test length(snr_det.rho_obs) == 2
    @test length(snr_src.rho_obs) == 2
    @test z_to_dl(0.2) > 0
    @test dl_to_z(z_to_dl(0.2)) ≈ 0.2 rtol=1e-8
    @test dvc_dz_fullsky(0.2) > 0

    mass_params = (alpha=2.0, beta=1.0, mmin=5.0, mmax=50.0)
    mass_draw = generate_mass_inj(MersenneTwister(11), 12, "PowerLaw", mass_params)
    @test length(mass_draw.mass_1_source) == 12
    @test all(mass_draw.mass_1_source .>= mass_draw.mass_2_source)
    @test all(>(0), mass_draw.prior)
    single_draw = generate_single_mass_inj(MersenneTwister(12), 8, "PowerLaw", mass_params)
    @test length(single_draw.mass_source) == 8
    @test all(5.0 .<= single_draw.mass_source .<= 50.0)
    peak_draw = generate_mass_inj(MersenneTwister(13), 6, "PowerLawPeak",
        (alpha=2.0, beta=1.0, mmin=5.0, mmax=50.0, mu_g=30.0, sigma_g=3.0, lambda_peak=0.2))
    @test length(peak_draw.prior) == 6
    multi_draw = generate_mass_inj(MersenneTwister(14), 6, "MultiPeak",
        (alpha=2.0, beta=1.0, mmin=5.0, mmax=50.0, mu_g_low=12.0, sigma_g_low=1.5,
            lambda_g_low=0.4, mu_g_high=35.0, sigma_g_high=3.0, lambda_g=0.2))
    @test length(multi_draw.prior) == 6
    dL_powerlaw = generate_dL_inj(MersenneTwister(15), 10, 0.5)
    @test length(dL_powerlaw.luminosity_distance) == 10
    @test all(>(0), dL_powerlaw.prior)
    @test length(generate_dL_inj_uniform(MersenneTwister(16), 5, 0.5).prior) == 5
    @test length(generate_dL_inj_z_uniform(MersenneTwister(17), 5, 0.5).redshift) == 5
    generated_inj = injection_set_generator(MersenneTwister(18), 4, 16, "PowerLaw", mass_params;
        zmax=0.5, snr_threshold=0.0, fgw_cut=0.0)
    @test length(generated_inj.prior) == 4
    @test generated_inj.ntotal_generated >= 16
    @test generated_inj.ndetected >= 4
    @test generated_inj.injections isa InjectionSet
    @test length(generated_inj.injections.prior) == 4

    m1_rw, m2_rw, z_rw, labels_rw = dvc_dz_reweight(MersenneTwister(8), [20.0, 30.0, 40.0], [10.0, 15.0, 20.0],
        [0.1, 0.2, 0.3]; extra=([1.0, 2.0, 3.0],))
    @test length(m1_rw) == length(m2_rw) == length(z_rw) == length(labels_rw) == 3
    @test all(in([1.0, 2.0, 3.0]), labels_rw)

    prep = quick_data_preparation(MersenneTwister(9), [30.0, 35.0, 40.0], [20.0, 24.0, 30.0], [0.1, 0.12, 0.14];
        theta=[0.8, 0.9, 1.0], reweight=false, snr_threshold=0.0, fgw_cut=0.0)
    @test prep.detected_indices == [1, 2, 3]
    @test length(prep.rho_obs) == 3
    pe = pe_quick_generation_samples(MersenneTwister(10), prep.mass_1_source, prep.mass_2_source, prep.redshift,
        prep.theta, prep.detected_indices, prep.rho_obs, prep.mass_ratio_obs, prep.chirp_mass_detector_obs,
        prep.theta_obs; Ninj=2, Nsamp=5, Ngen=80)
    @test length(pe.indices) == 2
    @test length(pe.posterior_samples[string(first(pe.indices))].mass_1_source) == 5
    @test PE_quick_generation_samples === pe_quick_generation_samples
end

@testset "utility helpers" begin
    posterior = (mass_1=[30.0, 31.0], mass_2=[20.0, 21.0])
    @test check_posterior_samples_and_prior(posterior, [1.0, 2.0]) === nothing
    @test_throws ArgumentError check_posterior_samples_and_prior((mass_1=[30.0], mass_2=[20.0, 21.0]), [1.0])
    @test_throws ArgumentError check_posterior_samples_and_prior(posterior, [1.0, 0.0])
    @test check_bounds_1D([0.0, 2.0], 0.5, 1.5) == [true, true]
    @test check_bounds_2D([1.0, 0.5], [0.5, 0.7], [0.1, NaN]) == [false, true]
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
    redshift_pair = RedshiftConditionalMassDistribution(redshift_mass, PowerLaw(5.0, 100.0, 1.0))
    pair_rate = CBCVanillaRate(c, redshift_pair, rate; R0=1)
    z_from_dl = redshift_at_luminosity_distance(c, dl)
    m1s = m1d / (1 + z_from_dl)
    m2s = m2d / (1 + z_from_dl)
    expected_pair = logpdf(redshift_pair, m1s, m2s, z_from_dl) + log(dvc_dz(c, z_from_dl)) -
        log(detector_to_source_jacobian(z_from_dl, c)) - log1p(z_from_dl)
    @test log_event_rate(pair_rate, m1d, m2d, dl, prior) ≈ expected_pair

    level = 0
    nside = healpix_level_to_nside(level)
    npix = 12 * nside^2
    uniq = [level_ipix_to_uniq(level, ipix) for ipix in 0:(npix - 1)]
    dngal = [10.0 * iz + pix for iz in 1:3, pix in 1:npix]
    bg_vals = fill(0.0, 3, npix)
    mktempdir() do dir
        catalog_path = joinpath(dir, "rate_catalog.h5")
        h5open(catalog_path, "w") do h
            group = create_group(h, "K")
            write(group, "mthr_moc_map", fill(30.0, npix))
            write(group, "uniq_moc_map", uniq)
            write(group, "z_grid", [0.1, 0.2, 0.4])
            subgroup = create_group(group, "weighted")
            attrs(subgroup)["band"] = "K-glade+"
            attrs(subgroup)["epsilon"] = 0.8
            write(subgroup, "vals_interpolant", permutedims(dngal))
            write(subgroup, "bg_vals_interpolant", permutedims(bg_vals))
        end
        catalog = IcarogwCatalog(catalog_path, "K", "weighted"; cosmology=c)
        row = get_NUNIQ_pixel(catalog, 0.2, 0.1)
        catalog_rate = CBCCatalogVanillaRate(catalog, c, ConditionalMassDistribution(mass, PowerLaw(5.0, 100.0, 1.0)), rate; Rgal=2.0)
        expected_catalog = logpdf(catalog_rate.mass_distribution, m1s, m2s) + Icarogw.Rates.log_rate(rate, z_from_dl) +
            log(dngal[2, row]) - log1p(z_from_dl) - log(detector_to_source_jacobian(z_from_dl, c)) + log(2.0)
        @test log_event_rate(catalog_rate, m1d, m2d, dl, row, prior) ≈ expected_catalog
        @test log_injection_rate(catalog_rate, m1d, m2d, dl, row, prior) ≈
              logpdf(catalog_rate.mass_distribution, m1s, m2s) + Icarogw.Rates.log_rate(rate, z_from_dl) +
              log(sum(dngal[2, :]) / npix) - log1p(z_from_dl) - log(detector_to_source_jacobian(z_from_dl, c)) + log(2.0)

        catalog_ps = PosteriorSamples((mass_1=[m1d], mass_2=[m2d], luminosity_distance=[dl],
            sky_indices=[row], prior=[1.0]))
        catalog_inj = InjectionSet((mass_1=[m1d], mass_2=[m2d], luminosity_distance=[dl],
            sky_indices=[row], prior=[1.0]); ntotal=10, Tobs=1)
        @test isfinite(loglikelihood(catalog_rate, PopulationData(PosteriorSampleSet(catalog_ps), catalog_inj)))

        skymap_rate = CBCCatalogSkyMapRate(catalog, c, rate; Rgal=3.0)
        @test log_event_rate(skymap_rate, dl, row, prior) ≈
              Icarogw.Rates.log_rate(rate, z_from_dl) + log(dngal[2, row]) - log1p(z_from_dl) -
              log(abs(ddl_dz(c, z_from_dl))) + log(3.0)
        @test isfinite(log_injection_rate(skymap_rate, dl, row, prior))
    end

    ps = PosteriorSamples((mass_1=[m1d], mass_ratio=[q], luminosity_distance=[dl], prior=[1.0]))
    inj = InjectionSet((mass_1=[m1d], mass_ratio=[q], luminosity_distance=[dl], prior=[1.0]); ntotal=10, Tobs=1)
    data = PopulationData(PosteriorSampleSet(ps), inj)
    @test isfinite(loglikelihood(CBCMass1Rate(c, mass, qprior, rate; R0=1), data))
    ps_pair = PosteriorSamples((mass_1=[m1d], mass_2=[m2d], luminosity_distance=[dl], prior=[1.0]))
    inj_pair = InjectionSet((mass_1=[m1d], mass_2=[m2d], luminosity_distance=[dl], prior=[1.0]); ntotal=10, Tobs=1)
    @test isfinite(loglikelihood(pair_rate, PopulationData(PosteriorSampleSet(ps_pair), inj_pair)))

    spin_prior = DefaultSpinPrior(2.0, 3.0, 0.5, 0.4)
    spin_rate = SpinWeightedRate(CBCVanillaRate(c, ConditionalMassDistribution(mass, PowerLaw(5.0, 100.0, 1.0)), rate; R0=1), spin_prior)
    spin_cols = (0.2, 0.3, 0.4, -0.2)
    @test log_event_rate(spin_rate, m1d, m2d, dl, spin_cols..., prior) ≈
        log_event_rate(spin_rate.base, m1d, m2d, dl, prior) + logpdf(spin_prior, spin_cols...)
    ps_spin = PosteriorSamples((mass_1=[m1d], mass_2=[m2d], luminosity_distance=[dl],
        chi_1=[spin_cols[1]], chi_2=[spin_cols[2]], cos_t_1=[spin_cols[3]], cos_t_2=[spin_cols[4]], prior=[1.0]))
    inj_spin = InjectionSet((mass_1=[m1d], mass_2=[m2d], luminosity_distance=[dl],
        chi_1=[spin_cols[1]], chi_2=[spin_cols[2]], cos_t_1=[spin_cols[3]], cos_t_2=[spin_cols[4]], prior=[1.0]);
        ntotal=10, Tobs=1)
    @test isfinite(loglikelihood(spin_rate, PopulationData(PosteriorSampleSet(ps_spin), inj_spin)))

    mass_spin_prior = EvolvingGaussianSpinPrior(0.15, 0.2, 0.002, 0.001, 0.6, 0.3)
    mass_spin_rate = SpinWeightedRate(spin_rate.base, mass_spin_prior)
    @test mass_spin_rate.spin_columns == (:chi_1, :chi_2, :cos_t_1, :cos_t_2, :mass_1_source, :mass_2_source)
    @test log_event_rate(mass_spin_rate, m1d, m2d, dl, spin_cols..., 30.0, 20.0, prior) ≈
        log_event_rate(mass_spin_rate.base, m1d, m2d, dl, prior) + logpdf(mass_spin_prior, spin_cols..., 30.0, 20.0)

    eff_prior = GaussianSpinPrior(0.0, 0.3, 0.4, 0.2, 0.1)
    eff_rate = SpinWeightedRate(CBCMass1Rate(c, mass, qprior, rate; R0=1), eff_prior)
    @test log_event_rate(eff_rate, m1d, q, dl, 0.1, 0.3, prior) ≈
        log_event_rate(eff_rate.base, m1d, q, dl, prior) + logpdf(eff_prior, 0.1, 0.3)

    pseob_prior = PSEOBGaussianPrior(0.1, 1.2, -0.2, 1.5, 0.25)
    pseob_rate = SpinWeightedRate(spin_rate.base, pseob_prior)
    @test pseob_rate.spin_columns == (:domega220, :dtau220)
    @test log_event_rate(pseob_rate, m1d, m2d, dl, -0.2, 0.4, prior) ≈
        log_event_rate(pseob_rate.base, m1d, m2d, dl, prior) + logpdf(pseob_prior, -0.2, 0.4)

    mixture = MixtureRate(
        CBCVanillaRate(c, ConditionalMassDistribution(mass, PowerLaw(5.0, 100.0, 1.0)), PowerLawRate(0.0); R0=1),
        CBCVanillaRate(c, ConditionalMassDistribution(mass, PowerLaw(5.0, 100.0, 1.0)), PowerLawRate(1.0); R0=1),
        0.35,
    )
    l1 = log_event_rate(mixture.rate1, m1d, m2d, dl, prior)
    l2 = log_event_rate(mixture.rate2, m1d, m2d, dl, prior)
    @test log_event_rate(mixture, m1d, m2d, dl, prior) ≈ logaddexp(log(0.35) + l1, log1p(-0.35) + l2)
    @test isfinite(loglikelihood(mixture, PopulationData(PosteriorSampleSet(ps_pair), inj_pair)))
end

@testset "planned placeholders" begin
    @test_throws ErrorException Icarogw.Catalog.catalog_planned()
    @test_throws ErrorException Icarogw.Stochastic.stochastic_planned()
    @test_throws ErrorException Icarogw.OmegaGW.omega_gw_planned()
end

@testset "stochastic omega gw" begin
    powers = pn_velocity_powers(60.0, 25.0)
    @test powers.v1 > 0
    @test powers.v2 ≈ powers.v1^2
    @test powers.v3 ≈ powers.v1^3

    rng = MersenneTwister(2026)
    freqs = [20.0, 40.0, 80.0]
    weights = precompute_omega_weights(rng, freqs; tmp_min=5.0, tmp_max=12.0, n=8, pn=false)
    @test weights.frequencies == freqs
    @test length(weights) == 8
    @test size(weights.dEdfs) == (8, 3)
    @test all(isfinite, weights.dEdfs)

    model = SimplePowerLawPopulation(
        cosmology=FlatLambdaCDM(H0=67.7, Om0=0.308, zmax=10),
        mass=ConditionalMassDistribution(PowerLaw(5.0, 12.0, -2.0), PowerLaw(5.0, 12.0, 1.0)),
        redshift_rate=PowerLawRate(0.0),
        R0=25.0,
    )
    omega, diag = spectral_siren_omega_gw(model, weights; return_diagnostics=true)
    @test length(omega) == length(freqs)
    @test all(isfinite, omega)
    @test all(>=(0), omega)
    @test diag.nweights == 8
    @test diag.nfrequencies == 3
    @test diag.max_weight >= diag.min_weight >= 0
    @test diag.has_nan == false

    residual = [1e-12, -2e-12, 3e-12]
    sigma2s = fill(4e-24, length(freqs))
    data = StochasticData(freqs, omega .+ residual, sigma2s; reference_H0=model.cosmology.H0)
    expected_stochastic = -0.5 * sum((abs.(omega .- data.Cf) .^ 2) ./ data.sigma2s)
    @test stochastic_loglikelihood(model, weights, data) ≈ expected_stochastic

    hscale = model.cosmology.H0 / 100.0
    scaled_data = StochasticData(freqs, (omega .+ residual) .* hscale^2, sigma2s .* hscale^4)
    @test stochastic_loglikelihood(model, weights, scaled_data) ≈ expected_stochastic

    mktempdir() do dir
        stochastic_csv = joinpath(dir, "stochastic.csv")
        CSV.write(stochastic_csv, DataFrame(freqs=freqs, Cf=data.Cf, sigma2s=data.sigma2s))
        data_from_csv = read_stochastic_csv(stochastic_csv; reference_H0=data.reference_H0)
        @test data_from_csv.frequencies == data.frequencies
        @test data_from_csv.Cf == data.Cf
        @test data_from_csv.sigma2s == data.sigma2s
        @test data_from_csv.reference_H0 == data.reference_H0

        stochastic_h5 = joinpath(dir, "stochastic.h5")
        write_stochastic_hdf5(stochastic_h5, data)
        data_from_h5 = read_stochastic_hdf5(stochastic_h5)
        @test data_from_h5.frequencies == data.frequencies
        @test data_from_h5.Cf == data.Cf
        @test data_from_h5.sigma2s == data.sigma2s
        @test data_from_h5.reference_H0 == data.reference_H0
        @test read_stochastic_hdf5(stochastic_h5; reference_H0=100.0).reference_H0 == 100.0
    end

    tiny = _tiny_population_data(FlatLambdaCDM(H0=67.7, Om0=0.308, zmax=2))
    cbc_logl = loglikelihood(model, tiny)
    @test joint_loglikelihood(model, tiny, weights, data) ≈ cbc_logl + expected_stochastic
    @test_throws ArgumentError StochasticData(freqs[1:2], omega, sigma2s)
    @test_throws ArgumentError StochasticData(freqs, omega, [-1.0, 1.0, 1.0])
    @test_throws ArgumentError stochastic_loglikelihood(model, weights, StochasticData([20.0, 41.0, 80.0], omega, sigma2s))
end

@testset "migration control files" begin
    repo = dirname(@__DIR__)
    goal_file = joinpath(repo, "2026-07-01-complete-python-science-features.md")
    audit_path = joinpath(repo, "docs", "migration_gap_audit.csv")
    audit_doc = joinpath(repo, "docs", "migration_gap_audit.md")
    fixture_script = joinpath(repo, "scripts", "fixtures", "generate_python_reference.py")
    integration_runner = joinpath(repo, "test", "integration", "runtests.jl")
    checklist = joinpath(repo, "docs", "reviews", "module_review_checklist.md")

    for path in (goal_file, audit_path, audit_doc, fixture_script, integration_runner, checklist)
        @test isfile(path)
    end

    audit = CSV.File(audit_path) |> DataFrame
    @test names(audit) == [
        "python_module",
        "python_api",
        "feature_area",
        "julia_target",
        "status",
        "fixture_priority",
        "next_phase",
        "notes",
    ]
    @test nrow(audit) >= 40
    @test Set(unique(audit.status)) ⊆ Set(["implemented", "partial", "missing", "excluded"])
    @test Set(unique(audit.fixture_priority)) ⊆ Set(["existing", "high", "medium", "low", "none"])
    @test all(!ismissing, audit.notes)

    modules = Set(audit.python_module)
    for mod in ("catalog.py", "stochastic.py", "omega_gw.py", "rates.py", "likelihood.py",
        "posterior_samples.py", "injections.py", "conversions.py", "priors.py", "wrappers.py",
        "simulation.py", "utils.py", "cupy_pal.py")
        @test mod in modules
    end

    @test any((audit.python_module .== "catalog.py") .& (audit.status .== "missing") .& (audit.fixture_priority .== "high"))
    @test any((audit.python_module .== "conversions.py") .& (audit.python_api .== "ligo_skymap") .& (audit.status .== "partial"))
    @test any((audit.python_module .== "conversions.py") .& occursin.("radec2skymap", audit.python_api) .& (audit.status .== "implemented"))
    @test any((audit.python_module .== "stochastic.py") .& (audit.status .== "implemented") .& (audit.fixture_priority .== "existing"))
    @test any((audit.python_module .== "omega_gw.py") .& (audit.status .== "implemented") .& (audit.fixture_priority .== "existing"))
    @test any((audit.python_module .== "likelihood.py") .& (audit.status .== "partial") .& (audit.next_phase .== "stochastic"))
    @test any((audit.python_module .== "utils.py") .& (audit.status .== "implemented") .& (audit.fixture_priority .== "existing"))
    @test any((audit.python_module .== "cupy_pal.py") .& (audit.status .== "implemented") .& (audit.fixture_priority .== "existing"))
    @test any((audit.python_module .== "cupy_pal.py") .& (audit.status .== "excluded"))
end

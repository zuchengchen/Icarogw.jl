module SkyMaps

using FITSIO
using Healpix
using Random
using StatsBase

export MOCMap,
    LigoSkyMap,
    ligo_skymap,
    healpix_level_to_nside,
    healpix_nside_to_level,
    uniq_to_level_ipix,
    level_ipix_to_uniq,
    radec2indeces,
    radec2indices,
    indices2radec,
    radec2skymap,
    pixel_area,
    intersect_em_pe!,
    intersect_em_pe,
    intersect_EM_PE,
    evaluate_3d_posterior_intersected,
    evaluate_3d_likelihood_intersected,
    evaluate_3d_posterior_likelihood,
    evaluate_3D_posterior_intersected,
    evaluate_3D_likelihood_intersected,
    evaluate_3D_posterior_likelihood,
    sample_3d_space,
    get_NUNIQ_pixel

"""
    healpix_level_to_nside(level)

Convert a LIGO multi-order HEALPix level to `nside`.
"""
healpix_level_to_nside(level::Integer) =
    level >= 0 ? 1 << Int(level) : throw(ArgumentError("HEALPix level must be non-negative"))

"""
    healpix_nside_to_level(nside)

Convert a power-of-two HEALPix `nside` to LIGO multi-order level.
"""
function healpix_nside_to_level(nside::Integer)
    n = Int(nside)
    n > 0 || throw(ArgumentError("nside must be positive"))
    ispow2(n) || throw(ArgumentError("nside must be a power of two for NUNIQ multi-order maps"))
    return trailing_zeros(n)
end

"""
    level_ipix_to_uniq(level, ipix)

Convert zero-based NESTED pixel index `ipix` at multi-order `level` to the
LIGO/IVOA NUNIQ identifier.
"""
function level_ipix_to_uniq(level::Integer, ipix::Integer)
    lev = Int(level)
    pix = Int(ipix)
    nside = healpix_level_to_nside(lev)
    0 <= pix < Healpix.nside2npix(nside) || throw(ArgumentError("ipix out of range for level $lev"))
    return 4 * nside^2 + pix
end

"""
    uniq_to_level_ipix(uniq)

Convert a LIGO/IVOA NUNIQ identifier to `(level, ipix)` where `ipix` is
zero-based in NESTED ordering.
"""
function uniq_to_level_ipix(uniq::Integer)
    u = Int(uniq)
    u >= 4 || throw(ArgumentError("NUNIQ identifier must be at least 4"))
    level = (floor(Int, log2(u ÷ 4))) ÷ 2
    nside = healpix_level_to_nside(level)
    ipix = u - 4 * nside^2
    0 <= ipix < Healpix.nside2npix(nside) || throw(ArgumentError("invalid NUNIQ identifier $uniq"))
    return level, ipix
end
uniq_to_level_ipix(uniq::AbstractArray) = begin
    pairs = map(uniq_to_level_ipix, uniq)
    first.(pairs), last.(pairs)
end

pixel_area(nside::Integer) = Healpix.nside2pixarea(Int(nside))

_as_float_vector(x::Real) = [float(x)]
_as_float_vector(x) = Float64.(collect(x))

function _resolve_resolution(nside::Integer)
    n = Int(nside)
    n > 0 || throw(ArgumentError("nside must be positive"))
    return Healpix.Resolution(n)
end

function _radec_to_index(ra::Real, dec::Real, nside::Integer; nest::Bool=false, zero_based::Bool=false)
    theta = pi / 2 - float(dec)
    phi = float(ra)
    res = _resolve_resolution(nside)
    pix = nest ? Healpix.ang2pixNest(res, theta, phi) : Healpix.ang2pixRing(res, theta, phi)
    return zero_based ? pix - 1 : pix
end

"""
    radec2indeces(ra, dec, nside; nest=false, zero_based=false)
    radec2indices(ra, dec, nside; nest=false, zero_based=false)

Convert right ascension and declination in radians to HEALPix pixel indices.
Julia returns 1-based indices by default; set `zero_based=true` for Python
`healpy` compatibility. The misspelled `radec2indeces` alias mirrors Python.
"""
function radec2indeces(ra::Real, dec::Real, nside::Integer; nest::Bool=false, zero_based::Bool=false)
    return _radec_to_index(ra, dec, nside; nest, zero_based)
end
function radec2indeces(ra, dec, nside::Integer; nest::Bool=false, zero_based::Bool=false)
    rav = _as_float_vector(ra)
    decv = _as_float_vector(dec)
    length(rav) == length(decv) || throw(ArgumentError("ra and dec must have the same length"))
    return [_radec_to_index(rav[i], decv[i], nside; nest, zero_based) for i in eachindex(rav)]
end
const radec2indices = radec2indeces

function _index_to_radec(index::Integer, nside::Integer; nest::Bool=false, zero_based::Bool=false)
    pix = zero_based ? Int(index) + 1 : Int(index)
    res = _resolve_resolution(nside)
    theta, phi = nest ? Healpix.pix2angNest(res, pix) : Healpix.pix2angRing(res, pix)
    return phi, pi / 2 - theta
end

"""
    indices2radec(indices, nside; nest=false, zero_based=false)

Convert HEALPix indices to pixel-center `(ra, dec)` in radians. Julia expects
1-based indices by default; set `zero_based=true` for Python `healpy`
compatibility.
"""
function indices2radec(index::Integer, nside::Integer; nest::Bool=false, zero_based::Bool=false)
    return _index_to_radec(index, nside; nest, zero_based)
end
function indices2radec(indices, nside::Integer; nest::Bool=false, zero_based::Bool=false)
    idx = Int.(collect(indices))
    ra = Vector{Float64}(undef, length(idx))
    dec = Vector{Float64}(undef, length(idx))
    for (i, pix) in pairs(idx)
        ra[i], dec[i] = _index_to_radec(pix, nside; nest, zero_based)
    end
    return ra, dec
end

"""
    radec2skymap(ra, dec, nside; nest=false)

Convert RA/Dec samples in radians to a normalized sky probability density map.
The returned tuple is `(counts_map, pixel_area_steradian)`, matching Python's
`radec2skymap` convention where `sum(counts_map) * pixel_area == 1`.
"""
function radec2skymap(ra, dec, nside::Integer; nest::Bool=false)
    rav = _as_float_vector(ra)
    decv = _as_float_vector(dec)
    length(rav) == length(decv) || throw(ArgumentError("ra and dec must have the same length"))
    !isempty(rav) || throw(ArgumentError("at least one sky sample is required"))
    npixels = Healpix.nside2npix(Int(nside))
    area = pixel_area(nside)
    counts = zeros(Float64, npixels)
    for pix in radec2indeces(rav, decv, nside; nest)
        counts[pix] += 1
    end
    counts ./= (length(rav) * area)
    return counts, area
end

"""
    MOCMap(values, uniq)

Minimal multi-order HEALPix map for NUNIQ-indexed catalog/skymap products.
`uniq` uses LIGO/IVOA NUNIQ identifiers and values are indexed by the map row.
"""
struct MOCMap{T}
    values::Vector{T}
    uniq::Vector{Int}
    levels::Vector{Int}
    ipix::Vector{Int}
    row_by_cell::Dict{Tuple{Int,Int},Int}
    max_level::Int
end

function MOCMap(values::AbstractVector{T}, uniq::AbstractVector) where {T}
    vals = collect(values)
    un = Int.(collect(uniq))
    length(vals) == length(un) || throw(ArgumentError("values and uniq must have the same length"))
    levels = Vector{Int}(undef, length(un))
    ipix = Vector{Int}(undef, length(un))
    rows = Dict{Tuple{Int,Int},Int}()
    maxlev = 0
    for (i, u) in pairs(un)
        lev, pix = uniq_to_level_ipix(u)
        levels[i] = lev
        ipix[i] = pix
        rows[(lev, pix)] = i
        maxlev = max(maxlev, lev)
    end
    return MOCMap{T}(vals, un, levels, ipix, rows, maxlev)
end

Base.length(m::MOCMap) = length(m.values)
Base.getindex(m::MOCMap, row::Integer) = m.values[Int(row)]
Base.getindex(m::MOCMap, rows::AbstractArray{<:Integer}) = m.values[Int.(rows)]

function _nested_parent(ipix::Integer, from_level::Integer, to_level::Integer)
    from_level >= to_level || throw(ArgumentError("from_level must be >= to_level"))
    shift = 2 * (Int(from_level) - Int(to_level))
    return Int(ipix) >> shift
end

function _row_for_radec(m::MOCMap, ra::Real, dec::Real)
    high_nside = healpix_level_to_nside(m.max_level)
    high_pix = _radec_to_index(ra, dec, high_nside; nest=true, zero_based=true)
    for level in m.max_level:-1:0
        pix = _nested_parent(high_pix, m.max_level, level)
        row = get(m.row_by_cell, (level, pix), nothing)
        row === nothing || return row
    end
    throw(ArgumentError("sky coordinate is not covered by this MOC map"))
end

"""
    get_NUNIQ_pixel(moc, ra, dec)

Return row indices in a `MOCMap` for RA/Dec positions in radians. This mirrors
the Python catalog behavior where later interpolants index columns by the MOC
row rather than by the raw NUNIQ value.
"""
function get_NUNIQ_pixel(m::MOCMap, ra::Real, dec::Real)
    return _row_for_radec(m, ra, dec)
end
function get_NUNIQ_pixel(m::MOCMap, ra, dec)
    rav = _as_float_vector(ra)
    decv = _as_float_vector(dec)
    length(rav) == length(decv) || throw(ArgumentError("ra and dec must have the same length"))
    return [_row_for_radec(m, rav[i], decv[i]) for i in eachindex(rav)]
end

mutable struct LigoSkyMap
    uniq::Vector{Int}
    probdensity::Vector{Float64}
    distmu::Vector{Float64}
    distsigma::Vector{Float64}
    moc::MOCMap{Float64}
    intersected::Bool
    dl_means::Vector{Float64}
    dl_sigmas::Vector{Float64}
    sky_prob_rad2::Vector{Float64}
    pixels_area::Vector{Float64}
    matched_rows::Vector{Int}
end

function LigoSkyMap(uniq, probdensity, distmu, distsigma)
    un = Int.(collect(uniq))
    prob = Float64.(collect(probdensity))
    mu = Float64.(collect(distmu))
    sig = Float64.(collect(distsigma))
    length(un) == length(prob) == length(mu) == length(sig) ||
        throw(ArgumentError("UNIQ, PROBDENSITY, DISTMU, and DISTSIGMA columns must have the same length"))
    moc = MOCMap(prob, un)
    return LigoSkyMap(un, prob, mu, sig, moc, false, Float64[], Float64[], Float64[], Float64[], Int[])
end

function _table_hdu(f)
    for hdu in f
        hdu isa FITSIO.TableHDU && return hdu
    end
    throw(ArgumentError("FITS file does not contain a binary table HDU"))
end

"""
    LigoSkyMap(path)
    ligo_skymap(path)

Read a LIGO multi-order FITS skymap with `UNIQ`, `PROBDENSITY`, `DISTMU`, and
`DISTSIGMA` table columns.
"""
function LigoSkyMap(path::AbstractString)
    FITSIO.FITS(path, "r") do f
        hdu = _table_hdu(f)
        cols = Set(FITSIO.colnames(hdu))
        required = ("UNIQ", "PROBDENSITY", "DISTMU", "DISTSIGMA")
        missing = [c for c in required if !(c in cols)]
        isempty(missing) || throw(ArgumentError("skymap FITS file is missing columns: $(join(missing, ", "))"))
        return LigoSkyMap(read(hdu, "UNIQ"), read(hdu, "PROBDENSITY"), read(hdu, "DISTMU"), read(hdu, "DISTSIGMA"))
    end
end
const ligo_skymap = LigoSkyMap

function _row_for_skymap(s::LigoSkyMap, ra::Real, dec::Real)
    return _row_for_radec(s.moc, ra, dec)
end

get_NUNIQ_pixel(s::LigoSkyMap, ra::Real, dec::Real) = _row_for_skymap(s, ra, dec)
get_NUNIQ_pixel(s::LigoSkyMap, ra, dec) = _rows_for_radec(s, ra, dec)

function _pixel_area_for_row(s::LigoSkyMap, row::Integer)
    level = s.moc.levels[Int(row)]
    return pixel_area(healpix_level_to_nside(level))
end

function _rows_for_radec(s::LigoSkyMap, ra, dec)
    rav = _as_float_vector(ra)
    decv = _as_float_vector(dec)
    length(rav) == length(decv) || throw(ArgumentError("ra and dec must have the same length"))
    return [_row_for_skymap(s, rav[i], decv[i]) for i in eachindex(rav)]
end

"""
    intersect_em_pe!(skymap, ra, dec)

Cache the skymap rows and distance parameters associated with RA/Dec samples.
"""
function intersect_em_pe!(s::LigoSkyMap, ra, dec)
    rows = _rows_for_radec(s, ra, dec)
    s.dl_means = s.distmu[rows]
    s.dl_sigmas = s.distsigma[rows]
    s.sky_prob_rad2 = s.probdensity[rows]
    s.pixels_area = [_pixel_area_for_row(s, row) for row in rows]
    s.matched_rows = rows
    s.intersected = true
    return s
end
const intersect_em_pe = intersect_em_pe!
const intersect_EM_PE = intersect_em_pe!

function _distance_density(dl, mu, sigma)
    sigma > 0 || return 0.0
    return inv((2pi * sigma^2)^2) * exp(-0.5 * ((dl - mu) / sigma)^2)
end

function _distance_density(dl::AbstractVector, mu::AbstractVector, sigma::AbstractVector)
    length(dl) == length(mu) == length(sigma) || throw(ArgumentError("distance arrays must have the same length"))
    return [_distance_density(dl[i], mu[i], sigma[i]) for i in eachindex(dl)]
end

"""
    evaluate_3d_posterior_intersected(skymap, dl)

Evaluate `p(dL, RA, Dec)` for samples cached by `intersect_em_pe!`.
"""
function evaluate_3d_posterior_intersected(s::LigoSkyMap, dl)
    s.intersected || throw(ArgumentError("call intersect_em_pe! before evaluating intersected skymap probabilities"))
    dlv = _as_float_vector(dl)
    pdl = _distance_density(dlv, s.dl_means, s.dl_sigmas)
    return s.sky_prob_rad2 .* pdl
end

"""
    evaluate_3d_likelihood_intersected(skymap, dl)

Evaluate the 3D localization likelihood using the Python convention that
divides by an isotropic sky prior and a `dL^2` distance prior.
"""
function evaluate_3d_likelihood_intersected(s::LigoSkyMap, dl)
    dlv = _as_float_vector(dl)
    posterior = evaluate_3d_posterior_intersected(s, dlv)
    return posterior .* s.pixels_area ./ (dlv .^ 2)
end

"""
    evaluate_3d_posterior_likelihood(skymap, dl, ra, dec)

Return `(posterior, likelihood)` at RA/Dec/distance samples.
"""
function evaluate_3d_posterior_likelihood(s::LigoSkyMap, dl, ra, dec)
    rows = _rows_for_radec(s, ra, dec)
    dlv = _as_float_vector(dl)
    length(dlv) == length(rows) || throw(ArgumentError("dl, ra, and dec must have the same length"))
    mu = s.distmu[rows]
    sig = s.distsigma[rows]
    psky = s.probdensity[rows]
    area = [_pixel_area_for_row(s, row) for row in rows]
    posterior = psky .* _distance_density(dlv, mu, sig)
    return posterior, posterior .* area ./ (dlv .^ 2)
end
const evaluate_3D_posterior_intersected = evaluate_3d_posterior_intersected
const evaluate_3D_likelihood_intersected = evaluate_3d_likelihood_intersected
const evaluate_3D_posterior_likelihood = evaluate_3d_posterior_likelihood

"""
    sample_3d_space(rng, skymap, nsamples)

Sample luminosity distance, RA, and Dec from a LIGO skymap. The returned
coordinates are in radians and distances are in Mpc.
"""
function sample_3d_space(rng::AbstractRNG, s::LigoSkyMap, nsamples::Integer)
    nsamples >= 0 || throw(ArgumentError("nsamples must be non-negative"))
    areas = [_pixel_area_for_row(s, row) for row in eachindex(s.uniq)]
    probs = s.probdensity .* areas
    total = sum(probs)
    total > 0 && isfinite(total) || throw(ArgumentError("skymap probability weights must sum to a positive finite value"))
    rows = sample(rng, collect(eachindex(s.uniq)), Weights(probs ./ total), nsamples; replace=true)
    dl = Vector{Float64}(undef, nsamples)
    ra = Vector{Float64}(undef, nsamples)
    dec = Vector{Float64}(undef, nsamples)
    for (i, row) in pairs(rows)
        level = s.moc.levels[row]
        nside = healpix_level_to_nside(level)
        pix = s.moc.ipix[row] + 1
        theta, phi = Healpix.pix2angNest(Healpix.Resolution(nside), pix)
        ra[i] = phi
        dec[i] = pi / 2 - theta
        mu = s.distmu[row]
        sigma = s.distsigma[row]
        if mu < 0
            dl[i] = 0.0
        elseif sigma == 0
            dl[i] = mu
        else
            draw = -1.0
            while draw <= 0
                draw = mu + sigma * randn(rng)
            end
            dl[i] = draw
        end
    end
    return dl, ra, dec
end
sample_3d_space(s::LigoSkyMap, nsamples::Integer; rng=Random.default_rng()) =
    sample_3d_space(rng, s, nsamples)

end

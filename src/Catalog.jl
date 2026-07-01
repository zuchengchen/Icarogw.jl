module Catalog

using ..Cosmology
using ..Conversions
using ..SkyMaps: MOCMap, radec2indeces
using HDF5
using QuadGK
using Random
using Statistics: mean

import ..SkyMaps: get_NUNIQ_pixel
import ..Cosmology: background_effective_galaxy_density,
    log_luminosity_function,
    log_luminosity_pdf,
    luminosity_function,
    luminosity_pdf,
    sample_luminosity_function

export LegacyGalaxyLuminosityFunction,
    IcarogwCatalog,
    GwcosmoCatalog,
    icarogw_catalog,
    gwcosmo_catalog,
    galaxy_MF_dep,
    KCorrection,
    DeprecatedKCorrection,
    kcorr,
    kcorr_dep,
    user_normal,
    get_NUNIQ_pixel,
    calc_mthr,
    calc_Mthr,
    effective_galaxy_number_interpolant,
    make_me_empty!,
    make_me_empty,
    em_likelihood_prior_differential_volume,
    EM_likelihood_prior_differential_volume,
    catalog_planned

const _LEGACY_GALAXY_BANDS = Set(("W1", "K", "bJ"))
const _MODERN_KCORR_BANDS = Set(("W1-glade+", "K-glade+", "bJ-glade+", "W1-upglade", "g-upglade", "r-upglade"))
const _DEPRECATED_KCORR_BANDS = Set(("W1", "K", "bJ"))

function _legacy_galaxy_band_parameters(band::AbstractString)
    if band == "W1"
        return (-28.0, -16.6, -24.09, -1.12, 1.45e-2 * 1e9)
    elseif band == "bJ"
        return (-22.0, -16.5, -19.66, -1.21, 1.61e-2 * 1e9)
    elseif band == "K"
        return (-27.0, -19.0, -23.39, -1.09, 1.16e-2 * 1e9)
    else
        throw(ArgumentError("unknown legacy galaxy luminosity-function band $band; expected one of $(join(sort(collect(_LEGACY_GALAXY_BANDS)), ", "))"))
    end
end

_little_h(c::Cosmology.AbstractCosmology) =
    hasproperty(c, :H0) ? getproperty(c, :H0) / 100 : _little_h(getproperty(c, :base))

_optional_float(value, name) =
    value === nothing ? throw(ArgumentError("$name must be provided when band is not set")) : float(value)

"""
    LegacyGalaxyLuminosityFunction(; band, cosmology=FlatLambdaCDM())
    LegacyGalaxyLuminosityFunction(; Mmin, Mmax, Mstar, alpha, phistar, cosmology=FlatLambdaCDM())

Dependency-light compatibility helper for Python `catalog.galaxy_MF_dep`. It
uses the legacy `W1`, `K`, and `bJ` band names and stores the observed
little-`h` shifted Schechter parameters. Prefer `GalaxyLuminosityFunction` for
new code.
"""
struct LegacyGalaxyLuminosityFunction
    band::String
    Mmin::Float64
    Mmax::Float64
    Mstar::Float64
    alpha::Float64
    phistar::Float64
    little_h::Float64
    Mminobs::Float64
    Mmaxobs::Float64
    Mstarobs::Float64
    phistarobs::Float64
    norm::Float64
    effective_epsilon::Union{Nothing,Float64}
    effective_delta_grid::Vector{Float64}
    effective_integral_grid::Vector{Float64}
end

function LegacyGalaxyLuminosityFunction(; band=nothing, Mmin=nothing, Mmax=nothing, Mstar=nothing,
    alpha=nothing, phistar=nothing, cosmology::Cosmology.AbstractCosmology=Cosmology.FlatLambdaCDM(),
    epsilon=nothing)
    label = band === nothing ? "" : String(band)
    params = band === nothing ?
        (_optional_float(Mmin, "Mmin"), _optional_float(Mmax, "Mmax"),
            _optional_float(Mstar, "Mstar"), _optional_float(alpha, "alpha"),
            _optional_float(phistar, "phistar")) :
        _legacy_galaxy_band_parameters(label)
    Mminv, Mmaxv, Mstarv, alphav, phistarv = params
    Mminv < Mmaxv || throw(ArgumentError("Mmin must be smaller than Mmax"))
    phistarv > 0 || throw(ArgumentError("phistar must be positive"))
    h = _little_h(cosmology)
    h > 0 || throw(ArgumentError("cosmology H0 must be positive"))
    hshift = 5log10(h)
    Mminobs = Mminv + hshift
    Mmaxobs = Mmaxv + hshift
    Mstarobs = Mstarv + hshift
    phistarobs = phistarv * h^3
    norm = _legacy_luminosity_norm(Mminobs, Mmaxobs, Mstarobs, alphav, phistarobs)
    eps = epsilon === nothing ? nothing : float(epsilon)
    delta_grid, integral_grid = eps === nothing ?
        (Float64[], Float64[]) :
        _legacy_effective_density_grid(Mminv, Mmaxv, Mstarv, alphav, eps)
    return LegacyGalaxyLuminosityFunction(label, Mminv, Mmaxv, Mstarv, alphav, phistarv, h,
        Mminobs, Mmaxobs, Mstarobs, phistarobs, norm, eps, delta_grid, integral_grid)
end
LegacyGalaxyLuminosityFunction(band::AbstractString; kwargs...) =
    LegacyGalaxyLuminosityFunction(; band, kwargs...)

const galaxy_MF_dep = LegacyGalaxyLuminosityFunction

function _legacy_luminosity_norm(Mminobs::Real, Mmaxobs::Real, Mstarobs::Real, alpha::Real, phistarobs::Real)
    value, _ = quadgk(M -> exp(_legacy_log_luminosity_unbounded(M, Mstarobs, alpha, phistarobs)),
        Mminobs, Mmaxobs; rtol=1e-8)
    return value
end

function _legacy_log_luminosity_unbounded(M::Real, Mstarobs::Real, alpha::Real, phistarobs::Real)
    x = 0.4 * (Mstarobs - M)
    return log(0.4 * log(10) * phistarobs) + (alpha + 1) * x * log(10) - 10.0^x
end

function _legacy_effective_density_integral(Mmin::Real, Mmax::Real, Mstar::Real, alpha::Real, epsilon::Real, Mthr::Real)
    xmin = 10.0^(0.4 * (Mstar - Mmax))
    xmax = 10.0^(0.4 * (Mstar - Mthr))
    xmax <= xmin && return 0.0
    value, _ = quadgk(x -> x^(alpha + epsilon) * exp(-x), xmin, xmax; rtol=1e-8)
    return value
end

function _legacy_effective_density_grid(Mmin::Real, Mmax::Real, Mstar::Real, alpha::Real, epsilon::Real)
    mgrid = collect(range(Mmin, Mmax; length=100))
    deltas = reverse(Mstar .- mgrid)
    integrals = reverse(map(M -> _legacy_effective_density_integral(Mmin, Mmax, Mstar, alpha, epsilon, M), mgrid))
    return deltas, integrals
end

function _interp_clamped(x::Real, xs::AbstractVector, ys::AbstractVector)
    length(xs) == length(ys) || throw(ArgumentError("interpolation arrays must have the same length"))
    isempty(xs) && throw(ArgumentError("interpolation arrays must not be empty"))
    x <= first(xs) && return first(ys)
    x >= last(xs) && return last(ys)
    hi = searchsortedfirst(xs, x)
    lo = hi - 1
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + t * (ys[hi] - ys[lo])
end

function log_luminosity_function(g::LegacyGalaxyLuminosityFunction, M::Real)
    g.Mminobs <= M <= g.Mmaxobs || return -Inf
    return _legacy_log_luminosity_unbounded(M, g.Mstarobs, g.alpha, g.phistarobs)
end
log_luminosity_function(g::LegacyGalaxyLuminosityFunction, M::AbstractArray) =
    map(m -> log_luminosity_function(g, m), M)
luminosity_function(g::LegacyGalaxyLuminosityFunction, M) = exp.(log_luminosity_function(g, M))

function log_luminosity_pdf(g::LegacyGalaxyLuminosityFunction, M::Real)
    lval = log_luminosity_function(g, M)
    isfinite(lval) || return -Inf
    return lval - log(g.norm)
end
log_luminosity_pdf(g::LegacyGalaxyLuminosityFunction, M::AbstractArray) =
    map(m -> log_luminosity_pdf(g, m), M)
luminosity_pdf(g::LegacyGalaxyLuminosityFunction, M) = exp.(log_luminosity_pdf(g, M))

function sample_luminosity_function(rng::AbstractRNG, g::LegacyGalaxyLuminosityFunction, n::Integer)
    grid = collect(range(g.Mminobs, g.Mmaxobs; length=10_000))
    weights = luminosity_pdf(g, grid)
    cdf = cumsum(weights)
    total = last(cdf)
    total > 0 && isfinite(total) || throw(ArgumentError("invalid legacy luminosity-function CDF"))
    cdf ./= total
    out = Vector{Float64}(undef, n)
    @inbounds for i in eachindex(out)
        u = rand(rng)
        j = searchsortedfirst(cdf, u)
        if j <= 1
            out[i] = first(grid)
        else
            t = (u - cdf[j - 1]) / (cdf[j] - cdf[j - 1])
            out[i] = grid[j - 1] + t * (grid[j] - grid[j - 1])
        end
    end
    return out
end
sample_luminosity_function(g::LegacyGalaxyLuminosityFunction, n::Integer; rng=Random.default_rng()) =
    sample_luminosity_function(rng, g, n)

function background_effective_galaxy_density(g::LegacyGalaxyLuminosityFunction, Mthr::Real)
    g.effective_epsilon === nothing && throw(ArgumentError(
        "construct LegacyGalaxyLuminosityFunction with epsilon to enable effective galaxy density",
    ))
    return g.phistarobs * _interp_clamped(g.Mstarobs - Mthr, g.effective_delta_grid, g.effective_integral_grid)
end
background_effective_galaxy_density(g::LegacyGalaxyLuminosityFunction, Mthr::AbstractArray) =
    map(M -> background_effective_galaxy_density(g, M), Mthr)

"""
    KCorrection(band)

Dependency-light K-correction formulas from Python `catalog.kcorr`. Supported
bands are `W1-glade+`, `K-glade+`, `bJ-glade+`, `W1-upglade`, `g-upglade`, and
`r-upglade`. The upGLADE bands use the local linearization
`k0 + dkbydz * (z - z0)`.
"""
struct KCorrection
    band::String
    function KCorrection(band::AbstractString)
        band in _MODERN_KCORR_BANDS || throw(ArgumentError(
            "unknown K-correction band $band; expected one of $(join(sort(collect(_MODERN_KCORR_BANDS)), ", "))",
        ))
        return new(String(band))
    end
end

const kcorr = KCorrection

_required_kcorr(value, name) =
    value === nothing ? throw(ArgumentError("$name must be provided for upGLADE K-corrections")) : value

function (k::KCorrection)(z::Real; k0=nothing, dkbydz=nothing, z0=nothing)
    if k.band == "W1-glade+"
        return -(4.44e-2 + 2.67z + 1.33z^2 - 1.59z^3)
    elseif k.band == "K-glade+"
        return -6.0 * log10(1 + z)
    elseif k.band == "bJ-glade+"
        return (z + 6z^2) / (1 + 20z^3)
    else
        return _required_kcorr(k0, "k0") + _required_kcorr(dkbydz, "dkbydz") * (z - _required_kcorr(z0, "z0"))
    end
end
(k::KCorrection)(z::AbstractArray; k0=nothing, dkbydz=nothing, z0=nothing) =
    k.band in ("W1-upglade", "g-upglade", "r-upglade") ?
    _required_kcorr(k0, "k0") .+ _required_kcorr(dkbydz, "dkbydz") .* (z .- _required_kcorr(z0, "z0")) :
    map(zi -> k(zi), z)

"""
    DeprecatedKCorrection(band)

Legacy K-correction formulas from Python `catalog.kcorr_dep`. Supported bands
are `W1`, `K`, and `bJ`.
"""
struct DeprecatedKCorrection
    band::String
    function DeprecatedKCorrection(band::AbstractString)
        band in _DEPRECATED_KCORR_BANDS || throw(ArgumentError(
            "unknown deprecated K-correction band $band; expected one of $(join(sort(collect(_DEPRECATED_KCORR_BANDS)), ", "))",
        ))
        return new(String(band))
    end
end

const kcorr_dep = DeprecatedKCorrection

function (k::DeprecatedKCorrection)(z::Real)
    if k.band == "W1"
        return -(4.44e-2 + 2.67z + 1.33z^2 - 1.59z^3)
    elseif k.band == "K"
        return -6.0 * log10(1 + z)
    else
        return (z + 6z^2) / (1 + 15z^3)
    end
end
(k::DeprecatedKCorrection)(z::AbstractArray) = map(zi -> k(zi), z)

"""
    user_normal(x, mu, sigma)

Normalized Gaussian density used by the Python catalog EM helper.
"""
function user_normal(x::Real, mu::Real, sigma::Real)
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    return inv(sqrt(2pi) * sigma) * exp(-0.5 * ((x - mu) / sigma)^2)
end
user_normal(x::AbstractArray, mu::Real, sigma::Real) = map(xi -> user_normal(xi, mu, sigma), x)

_cosmology_zmax(c::Cosmology.AbstractCosmology) =
    hasproperty(c, :zmax) ? getproperty(c, :zmax) : _cosmology_zmax(getproperty(c, :base))

function _trapz(y::AbstractVector, x::AbstractVector)
    length(y) == length(x) || throw(ArgumentError("trapz inputs must have the same length"))
    length(y) >= 2 || return 0.0
    total = 0.0
    @inbounds for i in 2:length(y)
        total += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    end
    return total
end

function _em_likelihood_scalar(z::Real, zobs::Real, sigmaz::Real, cosmology::Cosmology.AbstractCosmology,
    numsigma::Real, ptype::AbstractString)
    sigmaz > 0 || throw(ArgumentError("sigmaz must be positive"))
    zvalmin = max(1e-6, zobs - numsigma * sigmaz)
    zvalmax = min(zobs + 5sigmaz, _cosmology_zmax(cosmology))
    zvalmax <= zvalmin && return 0.0

    if ptype == "uniform"
        in_window = zobs - numsigma * sigmaz <= z <= zobs + numsigma * sigmaz
        in_window || return 0.0
        denom = Cosmology.comoving_volume(cosmology, zvalmax) - Cosmology.comoving_volume(cosmology, zvalmin)
        denom > 0 && isfinite(denom) || throw(ArgumentError("uniform EM prior normalization failed"))
        return 4pi * Cosmology.dvc_dz_dOmega(cosmology, z) / denom
    elseif ptype == "gaussian"
        zproxy = collect(range(zvalmin, zvalmax; length=5000))
        y = Cosmology.dvc_dz_dOmega(cosmology, zproxy) .* user_normal(zproxy, zobs, sigmaz)
        normfact = _trapz(y, zproxy)
        normfact > 0 && isfinite(normfact) || throw(ArgumentError("gaussian EM prior normalization failed"))
        return Cosmology.dvc_dz_dOmega(cosmology, z) * user_normal(z, zobs, sigmaz) / normfact
    elseif ptype == "gaussian_nocom"
        zproxy = collect(range(zvalmin, zvalmax; length=5000))
        y = user_normal(zproxy, zobs, sigmaz)
        normfact = _trapz(y, zproxy)
        normfact > 0 && isfinite(normfact) || throw(ArgumentError("gaussian_nocom EM prior normalization failed"))
        return user_normal(z, zobs, sigmaz) / normfact
    else
        throw(ArgumentError("unknown EM likelihood prior type $ptype; expected uniform, gaussian, or gaussian_nocom"))
    end
end

"""
    em_likelihood_prior_differential_volume(z, zobs, sigmaz, cosmology; Numsigma=3, ptype="uniform")

Redshift likelihood times prior helper from Python
`EM_likelihood_prior_differential_volume`. `ptype="uniform"` multiplies a
top-hat EM redshift likelihood by a uniform-in-comoving-volume prior;
`"gaussian"` uses a Gaussian EM likelihood with the same comoving-volume
factor; `"gaussian_nocom"` normalizes the Gaussian without the comoving-volume
factor.
"""
function em_likelihood_prior_differential_volume(z::Real, zobs::Real, sigmaz::Real,
    cosmology::Cosmology.AbstractCosmology; Numsigma::Real=3.0, ptype="uniform")
    return _em_likelihood_scalar(z, zobs, sigmaz, cosmology, Numsigma, String(ptype))
end
function em_likelihood_prior_differential_volume(z::AbstractArray, zobs::Real, sigmaz::Real,
    cosmology::Cosmology.AbstractCosmology; Numsigma::Real=3.0, ptype="uniform")
    ptype_string = String(ptype)
    sigmaz > 0 || throw(ArgumentError("sigmaz must be positive"))
    zvalmin = max(1e-6, zobs - Numsigma * sigmaz)
    zvalmax = min(zobs + 5sigmaz, _cosmology_zmax(cosmology))
    zvalmax <= zvalmin && return zeros(Float64, size(z))

    if ptype_string == "uniform"
        denom = Cosmology.comoving_volume(cosmology, zvalmax) - Cosmology.comoving_volume(cosmology, zvalmin)
        denom > 0 && isfinite(denom) || throw(ArgumentError("uniform EM prior normalization failed"))
        lo = zobs - Numsigma * sigmaz
        hi = zobs + Numsigma * sigmaz
        return map(zi -> lo <= zi <= hi ? 4pi * Cosmology.dvc_dz_dOmega(cosmology, zi) / denom : 0.0, z)
    elseif ptype_string == "gaussian"
        zproxy = collect(range(zvalmin, zvalmax; length=5000))
        y = Cosmology.dvc_dz_dOmega(cosmology, zproxy) .* user_normal(zproxy, zobs, sigmaz)
        normfact = _trapz(y, zproxy)
        normfact > 0 && isfinite(normfact) || throw(ArgumentError("gaussian EM prior normalization failed"))
        return map(zi -> Cosmology.dvc_dz_dOmega(cosmology, zi) * user_normal(zi, zobs, sigmaz) / normfact, z)
    elseif ptype_string == "gaussian_nocom"
        zproxy = collect(range(zvalmin, zvalmax; length=5000))
        y = user_normal(zproxy, zobs, sigmaz)
        normfact = _trapz(y, zproxy)
        normfact > 0 && isfinite(normfact) || throw(ArgumentError("gaussian_nocom EM prior normalization failed"))
        return map(zi -> user_normal(zi, zobs, sigmaz) / normfact, z)
    else
        throw(ArgumentError("unknown EM likelihood prior type $ptype_string; expected uniform, gaussian, or gaussian_nocom"))
    end
end

const EM_likelihood_prior_differential_volume = em_likelihood_prior_differential_volume

function _hdf5_attr_string(value)
    value isa AbstractString && return String(value)
    value isa AbstractVector{UInt8} && return String(value)
    return String(value)
end

function _read_matrix(group, name::AbstractString, nz::Integer, npix::Integer)
    raw = Array{Float64}(read(group, name))
    if size(raw) == (nz, npix)
        return raw
    elseif size(raw) == (npix, nz)
        return Matrix(permutedims(raw))
    else
        throw(ArgumentError("dataset $name has size $(size(raw)); expected ($nz, $npix) or Python/HDF5 transposed ($npix, $nz)"))
    end
end

function _interp_linear(x::Real, xs::AbstractVector, ys::AbstractVector; left=0.0, right=0.0)
    length(xs) == length(ys) || throw(ArgumentError("interpolation arrays must have the same length"))
    isempty(xs) && throw(ArgumentError("interpolation arrays must not be empty"))
    x < first(xs) && return left
    x > last(xs) && return right
    hi = searchsortedfirst(xs, x)
    if hi <= 1
        return first(ys)
    elseif hi > length(xs)
        return last(ys)
    elseif xs[hi] == x
        return ys[hi]
    else
        lo = hi - 1
        t = (x - xs[lo]) / (xs[hi] - xs[lo])
        return ys[lo] + t * (ys[hi] - ys[lo])
    end
end

function _interp_bounds(xs::AbstractVector, x::Real)
    (x < first(xs) || x > last(xs)) && return nothing
    hi = searchsortedfirst(xs, x)
    if hi <= 1
        return 1, 1, 0.0
    elseif hi > length(xs)
        return length(xs), length(xs), 0.0
    elseif xs[hi] == x
        return hi, hi, 0.0
    else
        lo = hi - 1
        return lo, hi, (x - xs[lo]) / (xs[hi] - xs[lo])
    end
end

function _interp2_linear(z_grid::AbstractVector, sky_grid::AbstractVector, values::AbstractMatrix, z::Real, skypos::Real; fill=0.0)
    zbounds = _interp_bounds(z_grid, z)
    sbounds = _interp_bounds(sky_grid, skypos)
    (zbounds === nothing || sbounds === nothing) && return fill
    zlo, zhi, zt = zbounds
    slo, shi, st = sbounds
    v00 = values[zlo, slo]
    v10 = values[zhi, slo]
    v01 = values[zlo, shi]
    v11 = values[zhi, shi]
    vlo = v00 + zt * (v10 - v00)
    vhi = v01 + zt * (v11 - v01)
    return vlo + st * (vhi - vlo)
end

function _vectorize_pair(z, skypos)
    z_is_scalar = z isa Real
    sky_is_scalar = skypos isa Real
    zv = z_is_scalar ? [float(z)] : Float64.(vec(collect(z)))
    sv = sky_is_scalar ? [Int(skypos)] : Int.(vec(collect(skypos)))
    if z_is_scalar && !sky_is_scalar
        zv = fill(first(zv), length(sv))
    elseif !z_is_scalar && sky_is_scalar
        sv = fill(first(sv), length(zv))
    elseif length(zv) != length(sv)
        throw(ArgumentError("z and sky position inputs must have the same length unless one is scalar"))
    end
    shape = z_is_scalar ? (sky_is_scalar ? nothing : size(skypos)) : size(z)
    return zv, sv, shape
end

_shape_output(values::Vector{Float64}, ::Nothing) = only(values)
_shape_output(values::Vector{Float64}, shape) = reshape(values, shape)

function _optional_distance_vector(dl, n::Integer)
    dl === nothing && return nothing
    values = dl isa Real ? fill(float(dl), n) : Float64.(vec(collect(dl)))
    length(values) == n || throw(ArgumentError("dl must have the same length as z"))
    return values
end

function _distance_vector(cosmology::Cosmology.AbstractCosmology, z::AbstractVector, dl)
    values = _optional_distance_vector(dl, length(z))
    return values === nothing ? Cosmology.luminosity_distance(cosmology, z) : values
end

"""
    IcarogwCatalog(path, grouping, subgrouping; cosmology=FlatLambdaCDM())

Runtime reader for Python `icarogw_catalog` HDF5 products. It expects
`mthr_moc_map`, `uniq_moc_map`, `z_grid`, and a subgroup containing
`vals_interpolant` and `bg_vals_interpolant`.
"""
struct IcarogwCatalog
    path::String
    grouping::String
    subgrouping::String
    moc_mthr_map::MOCMap{Float64}
    z_grid::Vector{Float64}
    band::String
    epsilon::Float64
    luminosity_function::Cosmology.GalaxyLuminosityFunction
    abs_magnitude_rate::Cosmology.LogPowerLawAbsMagnitudeRate
    sky_grid::Vector{Float64}
    dNgal_dzdOm_vals::Matrix{Float64}
    dNgal_dzdOm_vals_av::Vector{Float64}
    bg_vals_av::Vector{Float64}
end

function IcarogwCatalog(path::AbstractString, grouping::AbstractString, subgrouping::AbstractString;
    cosmology::Cosmology.AbstractCosmology=Cosmology.FlatLambdaCDM())
    h5open(path, "r") do h
        group = h[String(grouping)]
        subgroup = group[String(subgrouping)]
        mthr = Float64.(read(group, "mthr_moc_map"))
        uniq = Int.(read(group, "uniq_moc_map"))
        z_grid = Float64.(read(group, "z_grid"))
        band = _hdf5_attr_string(attrs(subgroup)["band"])
        epsilon = Float64(attrs(subgroup)["epsilon"])
        nz, npix = length(z_grid), length(uniq)
        dngal = _read_matrix(subgroup, "vals_interpolant", nz, npix)
        bg = _read_matrix(subgroup, "bg_vals_interpolant", nz, npix)
        lf = Cosmology.GalaxyLuminosityFunction(band; cosmology)
        rate = Cosmology.LogPowerLawAbsMagnitudeRate(epsilon)
        sky_grid = Float64.(1:npix)
        return IcarogwCatalog(String(path), String(grouping), String(subgrouping), MOCMap(mthr, uniq),
            z_grid, band, epsilon, lf, rate, sky_grid, dngal, vec(mean(dngal; dims=2)), vec(mean(bg; dims=2)))
    end
end

const icarogw_catalog = IcarogwCatalog

get_NUNIQ_pixel(c::IcarogwCatalog, ra, dec) = get_NUNIQ_pixel(c.moc_mthr_map, ra, dec)

"""
    calc_mthr(catalog, z, skypos, cosmology; dl=nothing)

Evaluate the absolute-magnitude threshold associated with catalog sky rows.
"""
function calc_mthr(c::IcarogwCatalog, z, skypos, cosmology::Cosmology.AbstractCosmology; dl=nothing)
    zv, sv, shape = _vectorize_pair(z, skypos)
    dlv = _distance_vector(cosmology, zv, dl)
    out = Vector{Float64}(undef, length(zv))
    @inbounds for i in eachindex(zv)
        row = sv[i]
        1 <= row <= length(c.moc_mthr_map) || throw(ArgumentError("catalog sky row $row outside 1:$(length(c.moc_mthr_map))"))
        out[i] = Conversions.absolute_magnitude(c.moc_mthr_map[row], dlv[i], 0.0)
    end
    return _shape_output(out, shape)
end
const calc_Mthr = calc_mthr

"""
    effective_galaxy_number_interpolant(catalog, z, skypos, cosmology; average=false, dl=nothing)

Evaluate the in-catalog and background effective galaxy terms
`(dNgal_dzdOmega, background)` for Python-compatible icarogw catalog files.
"""
function effective_galaxy_number_interpolant(c::IcarogwCatalog, z, skypos,
    cosmology::Cosmology.AbstractCosmology; average::Bool=false, dl=nothing)
    zv, sv, shape = _vectorize_pair(z, skypos)
    dlv = _distance_vector(cosmology, zv, dl)
    gc = Vector{Float64}(undef, length(zv))
    bg = Vector{Float64}(undef, length(zv))
    mthr = Float64.(vec(collect(calc_mthr(c, zv, sv, cosmology; dl=dlv))))
    @inbounds for i in eachindex(zv)
        zval = zv[i]
        row = sv[i]
        outside_grid = zval < first(c.z_grid) || zval > last(c.z_grid)
        if average
            gc[i] = _interp_linear(zval, c.z_grid, c.dNgal_dzdOm_vals_av; left=0.0, right=0.0)
            bg[i] = _interp_linear(zval, c.z_grid, c.bg_vals_av; left=first(c.bg_vals_av), right=last(c.bg_vals_av))
            if outside_grid
                bg[i] = background_effective_galaxy_density(c.luminosity_function, -Inf, zval, c.abs_magnitude_rate) *
                    Cosmology.dvc_dz_dOmega(cosmology, zval)
            end
        else
            gc[i] = _interp2_linear(c.z_grid, c.sky_grid, c.dNgal_dzdOm_vals, zval, row; fill=0.0)
            mthr_i = outside_grid ? -Inf : mthr[i]
            bg[i] = background_effective_galaxy_density(c.luminosity_function, mthr_i, zval, c.abs_magnitude_rate) *
                Cosmology.dvc_dz_dOmega(cosmology, zval)
        end
    end
    return _shape_output(gc, shape), _shape_output(bg, shape)
end

function make_me_empty!(c::IcarogwCatalog, cosmology::Cosmology.AbstractCosmology=Cosmology.FlatLambdaCDM(zmax=last(c.z_grid) * 2))
    c.dNgal_dzdOm_vals .= 0.0
    c.dNgal_dzdOm_vals_av .= 0.0
    c.bg_vals_av .= background_effective_galaxy_density(c.luminosity_function, -Inf, c.z_grid, c.abs_magnitude_rate) .*
        Cosmology.dvc_dz_dOmega(cosmology, c.z_grid)
    return c
end
const make_me_empty = make_me_empty!

function _gwcosmo_offset(group)
    opts = _hdf5_attr_string(attrs(group)["opts"])
    match_obj = match(r"['\"]offset['\"]\s*:\s*([-+0-9.eE]+)", opts)
    match_obj === nothing && throw(ArgumentError("gwcosmo catalog opts attribute does not contain an offset"))
    return parse(Float64, match_obj.captures[1])
end

function _gwcosmo_array(group, name::AbstractString, offset::Real)
    return exp.(Float64.(read(group, name))) .- offset
end

"""
    GwcosmoCatalog(path, nside, band, epsilon)

Runtime reader for gwcosmo-style line-of-sight HDF5 catalogs.
"""
struct GwcosmoCatalog
    path::String
    nside::Int
    band::String
    epsilon::Float64
    z_grid::Vector{Float64}
    pz_empty::Vector{Float64}
    dNgal_dzdOm_vals_av::Vector{Float64}
    dNgal_dzdOm_vals::Matrix{Float64}
    sky_grid::Vector{Float64}
    luminosity_function::Cosmology.GalaxyLuminosityFunction
    abs_magnitude_rate::Cosmology.LogPowerLawAbsMagnitudeRate
end

function GwcosmoCatalog(path::AbstractString, nside::Integer, band::AbstractString, epsilon::Real;
    cosmology::Cosmology.AbstractCosmology=Cosmology.FlatLambdaCDM())
    h5open(path, "r") do h
        offset = _gwcosmo_offset(h)
        z_grid = Float64.(read(h, "z_array"))
        npix = 12 * Int(nside)^2
        vals = Matrix{Float64}(undef, length(z_grid), npix)
        for pix in 0:(npix - 1)
            vals[:, pix + 1] = _gwcosmo_array(h, string(pix), offset)
        end
        return GwcosmoCatalog(String(path), Int(nside), String(band), Float64(epsilon),
            z_grid,
            _gwcosmo_array(h, "empty_catalogue", offset),
            _gwcosmo_array(h, "combined_pixels", offset),
            vals,
            Float64.(1:npix),
            Cosmology.GalaxyLuminosityFunction(String(band); cosmology),
            Cosmology.LogPowerLawAbsMagnitudeRate(epsilon))
    end
end

const gwcosmo_catalog = GwcosmoCatalog

get_NUNIQ_pixel(c::GwcosmoCatalog, ra, dec) = radec2indeces(ra, dec, c.nside; nest=true)

function effective_galaxy_number_interpolant(c::GwcosmoCatalog, z, skypos,
    cosmology::Cosmology.AbstractCosmology; average::Bool=false, dl=nothing)
    zv, sv, shape = _vectorize_pair(z, skypos)
    gc = Vector{Float64}(undef, length(zv))
    bg = zeros(Float64, length(zv))
    @inbounds for i in eachindex(zv)
        gc[i] = average ?
            _interp_linear(zv[i], c.z_grid, c.dNgal_dzdOm_vals_av; left=0.0, right=0.0) :
            _interp2_linear(c.z_grid, c.sky_grid, c.dNgal_dzdOm_vals, zv[i], sv[i]; fill=0.0)
    end
    return _shape_output(gc, shape), _shape_output(bg, shape)
end

function make_me_empty!(c::GwcosmoCatalog)
    c.dNgal_dzdOm_vals_av .= c.pz_empty
    for j in axes(c.dNgal_dzdOm_vals, 2)
        c.dNgal_dzdOm_vals[:, j] .= c.pz_empty
    end
    return c
end

"""
    catalog_planned()

Higher-level galaxy catalog, dark-siren, and bright-siren workflows are still
planned beyond the runtime catalog readers. This placeholder exists so users
get an explicit error instead of a silent partial implementation.
"""
function catalog_planned()
    throw(ErrorException("Higher-level catalog and EM-counterpart workflows are planned beyond the runtime catalog readers."))
end

end

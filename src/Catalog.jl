module Catalog

using ..Cosmology

export KCorrection,
    DeprecatedKCorrection,
    kcorr,
    kcorr_dep,
    user_normal,
    em_likelihood_prior_differential_volume,
    EM_likelihood_prior_differential_volume,
    catalog_planned

const _MODERN_KCORR_BANDS = Set(("W1-glade+", "K-glade+", "bJ-glade+", "W1-upglade", "g-upglade", "r-upglade"))
const _DEPRECATED_KCORR_BANDS = Set(("W1", "K", "bJ"))

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

"""
    catalog_planned()

Galaxy catalog, dark-siren, and bright-siren functionality is planned but not
implemented in the first native Julia version. This placeholder exists so users
get an explicit error instead of a silent partial implementation.
"""
function catalog_planned()
    throw(ErrorException("Catalog and EM-counterpart functionality is planned, not implemented in Icarogw.jl first-version scope."))
end

end

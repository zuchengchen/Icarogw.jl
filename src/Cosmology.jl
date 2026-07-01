module Cosmology

using QuadGK
using Random
using Roots

export COST_C,
    AbstractCosmology,
    FlatLambdaCDM,
    FlatwCDM,
    Flatw0waCDM,
    Epsilon0Cosmology,
    Xi0Cosmology,
    ExtraDCosmology,
    PlanckMassCosmology,
    AlphaLogCosmology,
    luminosity_distance,
    redshift_at_luminosity_distance,
    comoving_distance,
    comoving_volume,
    dvc_dz,
    dvc_dz_dOmega,
    ddl_dz,
    sample_comoving_volume,
    GalaxyLuminosityFunction,
    galaxy_MF,
    AbstractAbsMagnitudeRate,
    basic_absM_rate,
    LogPowerLawAbsMagnitudeRate,
    log_powerlaw_absM_rate,
    evolved_luminosity_parameters,
    luminosity_function_norm,
    log_luminosity_function,
    luminosity_function,
    log_luminosity_pdf,
    luminosity_pdf,
    log_abs_magnitude_rate,
    abs_magnitude_rate,
    background_effective_galaxy_density,
    sample_luminosity_function

const COST_C = 299792.458

abstract type AbstractCosmology end

"""
    FlatLambdaCDM(; H0=67.7, Om0=0.308, zmax=10.0)

Flat `ΛCDM` cosmology with Hubble constant `H0` in km s^-1 Mpc^-1 and matter
density `Om0`. Distances returned by this package are in Mpc unless the
docstring explicitly says Gpc.
"""
Base.@kwdef struct FlatLambdaCDM <: AbstractCosmology
    H0::Float64 = 67.7
    Om0::Float64 = 0.308
    zmax::Float64 = 10.0
end

"""
    FlatwCDM(; H0=67.7, Om0=0.308, w0=-1, zmax=10)

Flat dark-energy cosmology with constant equation-of-state parameter `w0`.
Setting `w0=-1` recovers `FlatLambdaCDM`.
"""
Base.@kwdef struct FlatwCDM <: AbstractCosmology
    H0::Float64 = 67.7
    Om0::Float64 = 0.308
    w0::Float64 = -1.0
    zmax::Float64 = 10.0
end

"""
    Flatw0waCDM(; H0=67.7, Om0=0.308, w0=-1, wa=0, zmax=10)

Flat CPL dark-energy cosmology with `w(z) = w0 + wa*z/(1+z)`.
"""
Base.@kwdef struct Flatw0waCDM <: AbstractCosmology
    H0::Float64 = 67.7
    Om0::Float64 = 0.308
    w0::Float64 = -1.0
    wa::Float64 = 0.0
    zmax::Float64 = 10.0
end

"""
    Epsilon0Cosmology(base, eps0)

Modified-gravity luminosity distance wrapper using `dL_gw = (1 + z)^eps0 dL_em`.
Comoving volume remains that of the electromagnetic background cosmology.
"""
struct Epsilon0Cosmology{C<:AbstractCosmology} <: AbstractCosmology
    base::C
    eps0::Float64
end

"""
    Xi0Cosmology(base, Xi0, n)

Modified-gravity luminosity distance wrapper using
`dL_gw = dL_em * (Xi0 + (1 - Xi0) * (1 + z)^(-n))`.
"""
struct Xi0Cosmology{C<:AbstractCosmology} <: AbstractCosmology
    base::C
    Xi0::Float64
    n::Float64
end

"""
    ExtraDCosmology(base, D, n, Rc)

Extra-dimension phenomenological wrapper following the Python `extraD` model.
`Rc` is in Mpc because it multiplies luminosity distance.
"""
struct ExtraDCosmology{C<:AbstractCosmology} <: AbstractCosmology
    base::C
    D::Float64
    n::Float64
    Rc::Float64
end

"""
    PlanckMassCosmology(base, cM)

Running Planck-mass luminosity-distance wrapper. The integral is evaluated
against the background `FlatLambdaCDM` expansion.
"""
struct PlanckMassCosmology{C<:AbstractCosmology} <: AbstractCosmology
    base::C
    cM::Float64
end

"""
    AlphaLogCosmology(base, a1, a2, a3)

Log-polynomial luminosity-distance wrapper
`dL_gw = dL_em * (1 + a1 log(1+z) + a2 log(1+z)^2 + a3 log(1+z)^3)`.
"""
struct AlphaLogCosmology{C<:AbstractCosmology} <: AbstractCosmology
    base::C
    a1::Float64
    a2::Float64
    a3::Float64
end

zmax(c::Union{FlatLambdaCDM,FlatwCDM,Flatw0waCDM}) = c.zmax
zmax(c::Union{Epsilon0Cosmology,Xi0Cosmology,ExtraDCosmology,PlanckMassCosmology,AlphaLogCosmology}) = zmax(c.base)

efunc(c::FlatLambdaCDM, z::Real) = sqrt(c.Om0 * (1 + z)^3 + (1 - c.Om0))
efunc(c::FlatwCDM, z::Real) = sqrt(c.Om0 * (1 + z)^3 + (1 - c.Om0) * (1 + z)^(3 * (1 + c.w0)))
efunc(c::Flatw0waCDM, z::Real) = sqrt(c.Om0 * (1 + z)^3 + (1 - c.Om0) * (1 + z)^(3 * (1 + c.w0 + c.wa)) * exp(-3 * c.wa * z / (1 + z)))
hubble(c::Union{FlatLambdaCDM,FlatwCDM,Flatw0waCDM}, z::Real) = c.H0 * efunc(c, z)

function _check_z(c::AbstractCosmology, z::Real)
    0 <= z <= zmax(c) || throw(ArgumentError("redshift $z outside supported range [0, $(zmax(c))]"))
    return Float64(z)
end

function _check_dl(c::AbstractCosmology, dl::Real)
    dl >= 0 || throw(ArgumentError("luminosity distance must be non-negative"))
    dlmax = luminosity_distance(c, zmax(c))
    dl <= dlmax || throw(ArgumentError("luminosity distance $dl Mpc exceeds zmax=$(zmax(c)) distance $dlmax Mpc"))
    return Float64(dl)
end

"""
    comoving_distance(cosmology, z)

Line-of-sight comoving distance in Mpc for redshift `z`.
"""
function comoving_distance(c::Union{FlatLambdaCDM,FlatwCDM,Flatw0waCDM}, z::Real)
    z = _check_z(c, z)
    z == 0 && return 0.0
    val, _ = quadgk(x -> COST_C / hubble(c, x), 0.0, z; rtol=1e-9)
    return val
end
comoving_distance(c::Union{Epsilon0Cosmology,Xi0Cosmology,ExtraDCosmology,PlanckMassCosmology,AlphaLogCosmology}, z::Real) =
    comoving_distance(c.base, z)
comoving_distance(c::AbstractCosmology, z::AbstractArray) = map(x -> comoving_distance(c, x), z)

"""
    luminosity_distance(cosmology, z)

Luminosity distance in Mpc. Modified-gravity wrappers alter this value but keep
the background comoving volume from their base cosmology.
"""
luminosity_distance(c::Union{FlatLambdaCDM,FlatwCDM,Flatw0waCDM}, z::Real) = (1 + _check_z(c, z)) * comoving_distance(c, z)
luminosity_distance(c::Epsilon0Cosmology, z::Real) = luminosity_distance(c.base, z) * (1 + z)^c.eps0
luminosity_distance(c::Xi0Cosmology, z::Real) = luminosity_distance(c.base, z) * (c.Xi0 + (1 - c.Xi0) * (1 + z)^(-c.n))
function luminosity_distance(c::ExtraDCosmology, z::Real)
    dl = luminosity_distance(c.base, z)
    return dl * (1 + (dl / ((1 + z) * c.Rc))^c.n)^((c.D - 4) / (2c.n))
end
function luminosity_distance(c::PlanckMassCosmology, z::Real)
    z = _check_z(c, z)
    integral, _ = quadgk(x -> 1 / ((1 + x) * efunc(c.base, x)^2), 0.0, z; rtol=1e-8)
    return luminosity_distance(c.base, z) * exp(0.5 * c.cM * integral)
end
function luminosity_distance(c::AlphaLogCosmology, z::Real)
    l = log1p(_check_z(c, z))
    return luminosity_distance(c.base, z) * (1 + c.a1 * l + c.a2 * l^2 + c.a3 * l^3)
end
luminosity_distance(c::AbstractCosmology, z::AbstractArray) = map(x -> luminosity_distance(c, x), z)

"""
    redshift_at_luminosity_distance(cosmology, dl)

Invert `luminosity_distance(cosmology, z)` for luminosity distance `dl` in Mpc.
"""
function redshift_at_luminosity_distance(c::AbstractCosmology, dl::Real)
    dl = _check_dl(c, dl)
    dl == 0 && return 0.0
    return find_zero(z -> luminosity_distance(c, z) - dl, (0.0, zmax(c)), Bisection(); xtol=1e-10)
end
redshift_at_luminosity_distance(c::AbstractCosmology, dl::AbstractArray) =
    map(x -> redshift_at_luminosity_distance(c, x), dl)

"""
    comoving_volume(cosmology, z)

Full-sky comoving volume in Gpc^3 at redshift `z`.
"""
function comoving_volume(c::AbstractCosmology, z::Real)
    dc = comoving_distance(c, z)
    return 4pi / 3 * (dc / 1000)^3
end
comoving_volume(c::AbstractCosmology, z::AbstractArray) = map(x -> comoving_volume(c, x), z)

"""
    dvc_dz_dOmega(cosmology, z)

Differential comoving volume per steradian in Gpc^3 sr^-1.
"""
function dvc_dz_dOmega(c::AbstractCosmology, z::Real)
    bg = c isa Union{FlatLambdaCDM,FlatwCDM,Flatw0waCDM} ? c : c.base
    z = _check_z(bg, z)
    dm = comoving_distance(bg, z)
    return (COST_C / hubble(bg, z)) * dm^2 / 1e9
end
dvc_dz_dOmega(c::AbstractCosmology, z::AbstractArray) = map(x -> dvc_dz_dOmega(c, x), z)

"""
    dvc_dz(cosmology, z)

Full-sky differential comoving volume in Gpc^3 per unit redshift.
"""
dvc_dz(c::AbstractCosmology, z::Real) = 4pi * dvc_dz_dOmega(c, z)
dvc_dz(c::AbstractCosmology, z::AbstractArray) = map(x -> dvc_dz(c, x), z)

"""
    ddl_dz(cosmology, z)

Derivative of luminosity distance with respect to redshift, in Mpc.
"""
function ddl_dz(c::Union{FlatLambdaCDM,FlatwCDM,Flatw0waCDM}, z::Real)
    z = _check_z(c, z)
    return luminosity_distance(c, z) / (1 + z) + COST_C * (1 + z) / hubble(c, z)
end
function ddl_dz(c::AbstractCosmology, z::Real)
    z = _check_z(c, z)
    dz = max(1e-6, 1e-5 * (1 + z))
    lo = max(0.0, z - dz)
    hi = min(zmax(c), z + dz)
    hi == lo && return NaN
    return abs((luminosity_distance(c, hi) - luminosity_distance(c, lo)) / (hi - lo))
end
ddl_dz(c::AbstractCosmology, z::AbstractArray) = map(x -> ddl_dz(c, x), z)

"""
    sample_comoving_volume(rng, cosmology, n; zmin=0, zmax=cosmology.zmax)

Draw redshifts approximately uniform in comoving volume using a tabulated CDF.
"""
function sample_comoving_volume(rng::AbstractRNG, c::AbstractCosmology, n::Integer; zmin=0.0, zmax=zmax(c))
    grid = collect(range(zmin, zmax; length=4096))
    weights = dvc_dz(c, grid)
    cdf = cumsum(weights)
    cdf ./= last(cdf)
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
sample_comoving_volume(c::AbstractCosmology, n::Integer; kwargs...) =
    sample_comoving_volume(Random.default_rng(), c, n; kwargs...)

_little_h(c::Union{FlatLambdaCDM,FlatwCDM,Flatw0waCDM}) = c.H0 / 100
_little_h(c::Union{Epsilon0Cosmology,Xi0Cosmology,ExtraDCosmology,PlanckMassCosmology,AlphaLogCosmology}) =
    _little_h(c.base)

function _galaxy_band_parameters(band::AbstractString)
    if band == "W1-glade+" || band == "W1-upglade"
        return (-28.0, -16.6, -24.09, -1.12, 1.45e-2 * 1e9, 0.0, 0.0, 0.1)
    elseif band == "bJ-glade+"
        return (-22.0, -16.5, -19.66, -1.21, 1.61e-2 * 1e9, 0.0, 0.0, 0.1)
    elseif band == "K-glade+"
        return (-27.0, -19.0, -23.39, -1.09, 1.16e-2 * 1e9, 0.0, 0.0, 0.1)
    elseif band == "g-upglade"
        return (-24.0, -17.0, -20.14, -1.07, 1.62e-2 * 1e9, 0.0, 0.0, 0.1)
    elseif band == "r-upglade"
        return (-24.0, -17.0, -20.64, -1.09, 1.43e-2 * 1e9, 0.0, 0.0, 0.1)
    else
        throw(ArgumentError("unknown galaxy luminosity-function band: $band"))
    end
end

_required_float(value, name) =
    value === nothing ? throw(ArgumentError("$name must be provided when band is not set")) : float(value)

"""
    GalaxyLuminosityFunction(; band, cosmology=FlatLambdaCDM())
    GalaxyLuminosityFunction(; Mmin, Mmax, Mstar, alpha, phistar, Q=0, P=0, z0=0.1, cosmology=FlatLambdaCDM())

Schechter luminosity function in absolute magnitude, equivalent to Python
`galaxy_MF` after `build_MF`. `phistar` is in `Gpc^-3`. The observed absolute
magnitude parameters are shifted by `5log10(H0/100)`, matching the Python
`little_h` convention.
"""
struct GalaxyLuminosityFunction
    band::String
    Mmin::Float64
    Mmax::Float64
    Mstar::Float64
    alpha::Float64
    phistar::Float64
    Q::Float64
    P::Float64
    z0::Float64
    little_h::Float64
    Mminobs::Float64
    Mmaxobs::Float64
    Mstarobs::Float64
    phistarobs::Float64
end

function GalaxyLuminosityFunction(; band=nothing, Mmin=nothing, Mmax=nothing, Mstar=nothing,
    alpha=nothing, phistar=nothing, Q=0.0, P=0.0, z0=0.1,
    cosmology::AbstractCosmology=FlatLambdaCDM())
    label = band === nothing ? "" : String(band)
    params = band === nothing ?
        (_required_float(Mmin, "Mmin"), _required_float(Mmax, "Mmax"), _required_float(Mstar, "Mstar"),
            _required_float(alpha, "alpha"), _required_float(phistar, "phistar"),
            float(Q), float(P), float(z0)) :
        _galaxy_band_parameters(label)
    Mminv, Mmaxv, Mstarv, alphav, phistarv, Qv, Pv, z0v = params
    Mminv < Mmaxv || throw(ArgumentError("Mmin must be smaller than Mmax"))
    phistarv > 0 || throw(ArgumentError("phistar must be positive"))
    h = _little_h(cosmology)
    h > 0 || throw(ArgumentError("cosmology H0 must be positive"))
    hshift = 5 * log10(h)
    return GalaxyLuminosityFunction(label, Mminv, Mmaxv, Mstarv, alphav, phistarv, Qv, Pv, z0v,
        h, Mminv + hshift, Mmaxv + hshift, Mstarv + hshift, phistarv * h^3)
end
GalaxyLuminosityFunction(band::AbstractString; kwargs...) = GalaxyLuminosityFunction(; band, kwargs...)
const galaxy_MF = GalaxyLuminosityFunction

"""
    evolved_luminosity_parameters(galaxy_lf, z)

Return `(phistar(z), Mstar(z))` for the Schechter function evolution.
"""
function evolved_luminosity_parameters(g::GalaxyLuminosityFunction, z::Real)
    phistar = g.phistarobs * 10.0^(0.4 * g.P * z)
    Mstar = g.Mstarobs - g.Q * (z - g.z0)
    return phistar, Mstar
end
evolved_luminosity_parameters(g::GalaxyLuminosityFunction, z::AbstractArray) =
    map(v -> evolved_luminosity_parameters(g, v), z)

function _log_luminosity_function_unbounded(g::GalaxyLuminosityFunction, M::Real, z::Real)
    phistar, Mstar = evolved_luminosity_parameters(g, z)
    x = 0.4 * (Mstar - M)
    return log(0.4 * log(10) * phistar) + (g.alpha + 1) * x * log(10) - 10.0^x
end

"""
    log_luminosity_function(galaxy_lf, M, z)

Log Schechter function value in `Gpc^-3 mag^-1`. Values outside the configured
absolute-magnitude interval return `-Inf`.
"""
function log_luminosity_function(g::GalaxyLuminosityFunction, M::Real, z::Real)
    g.Mminobs <= M <= g.Mmaxobs || return -Inf
    return _log_luminosity_function_unbounded(g, M, z)
end
log_luminosity_function(g::GalaxyLuminosityFunction, M::AbstractArray, z::AbstractArray) =
    map((m, zi) -> log_luminosity_function(g, m, zi), M, z)
log_luminosity_function(g::GalaxyLuminosityFunction, M::AbstractArray, z::Real) =
    map(m -> log_luminosity_function(g, m, z), M)
log_luminosity_function(g::GalaxyLuminosityFunction, M::Real, z::AbstractArray) =
    map(zi -> log_luminosity_function(g, M, zi), z)

luminosity_function(g::GalaxyLuminosityFunction, M, z) = exp.(log_luminosity_function(g, M, z))

"""
    luminosity_function_norm(galaxy_lf, z)

Integral of the Schechter function over the configured magnitude range. Julia
uses direct quadrature, which remains finite for common catalog bands where the
Python incomplete-gamma branch can return `NaN`.
"""
function luminosity_function_norm(g::GalaxyLuminosityFunction, z::Real)
    value, _ = quadgk(M -> exp(_log_luminosity_function_unbounded(g, M, z)),
        g.Mminobs, g.Mmaxobs; rtol=1e-8)
    return value
end
luminosity_function_norm(g::GalaxyLuminosityFunction, z::AbstractArray) =
    map(zi -> luminosity_function_norm(g, zi), z)

function log_luminosity_pdf(g::GalaxyLuminosityFunction, M::Real, z::Real)
    lval = log_luminosity_function(g, M, z)
    isfinite(lval) || return -Inf
    return lval - log(luminosity_function_norm(g, z))
end
log_luminosity_pdf(g::GalaxyLuminosityFunction, M::AbstractArray, z::AbstractArray) =
    map((m, zi) -> log_luminosity_pdf(g, m, zi), M, z)
log_luminosity_pdf(g::GalaxyLuminosityFunction, M::AbstractArray, z::Real) =
    map(m -> log_luminosity_pdf(g, m, z), M)
log_luminosity_pdf(g::GalaxyLuminosityFunction, M::Real, z::AbstractArray) =
    map(zi -> log_luminosity_pdf(g, M, zi), z)

luminosity_pdf(g::GalaxyLuminosityFunction, M, z) = exp.(log_luminosity_pdf(g, M, z))

"""
    sample_luminosity_function(rng, galaxy_lf, n, z)

Draw absolute magnitudes from the normalized Schechter function at redshift
`z` using a dense inverse-CDF table.
"""
function sample_luminosity_function(rng::AbstractRNG, g::GalaxyLuminosityFunction, n::Integer, z::Real)
    grid = collect(range(g.Mminobs, g.Mmaxobs; length=10_000))
    weights = luminosity_pdf(g, grid, z)
    cdf = cumsum(weights)
    total = last(cdf)
    total > 0 && isfinite(total) || throw(ArgumentError("invalid luminosity-function CDF"))
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
sample_luminosity_function(g::GalaxyLuminosityFunction, n::Integer, z::Real; rng=Random.default_rng()) =
    sample_luminosity_function(rng, g, n, z)

abstract type AbstractAbsMagnitudeRate end
const basic_absM_rate = AbstractAbsMagnitudeRate

"""
    LogPowerLawAbsMagnitudeRate(epsilon)

Absolute-magnitude CBC rate weight matching Python `log_powerlaw_absM_rate`.
Galaxies fainter than the Schechter faint limit have zero rate; brighter
outliers are retained, matching the Python catalog behavior.
"""
struct LogPowerLawAbsMagnitudeRate <: AbstractAbsMagnitudeRate
    epsilon::Float64
end
LogPowerLawAbsMagnitudeRate(epsilon::Real) = LogPowerLawAbsMagnitudeRate(float(epsilon))
const log_powerlaw_absM_rate = LogPowerLawAbsMagnitudeRate

function log_abs_magnitude_rate(r::LogPowerLawAbsMagnitudeRate, g::GalaxyLuminosityFunction, M::Real)
    M > g.Mmaxobs && return -Inf
    return r.epsilon * 0.4 * (g.Mstarobs - M) * log(10)
end
log_abs_magnitude_rate(r::LogPowerLawAbsMagnitudeRate, g::GalaxyLuminosityFunction, M::AbstractArray) =
    map(m -> log_abs_magnitude_rate(r, g, m), M)
abs_magnitude_rate(r::AbstractAbsMagnitudeRate, g::GalaxyLuminosityFunction, M) =
    exp.(log_abs_magnitude_rate(r, g, M))

function _effective_density_integral(g::GalaxyLuminosityFunction, delta::Real, epsilon::Real)
    delta_lo = g.Mstar - g.Mmax
    delta_hi = g.Mstar - g.Mmin
    delta <= delta_lo && return 0.0
    clipped = min(delta, delta_hi)
    xmin = 10.0^(0.4 * delta_lo)
    xmax = 10.0^(0.4 * clipped)
    xmax <= xmin && return 0.0
    shape = g.alpha + 1 + epsilon
    value, _ = quadgk(x -> x^(shape - 1) * exp(-x), xmin, xmax; rtol=1e-8)
    return value
end

"""
    background_effective_galaxy_density(galaxy_lf, Mthr, z; epsilon)

Effective background galaxy density `dN_eff/dVc`, integrating the Schechter
function from the faint-end limit to an absolute-magnitude threshold with the
luminosity weight exponent `epsilon`.
"""
function background_effective_galaxy_density(g::GalaxyLuminosityFunction, Mthr::Real, z::Real; epsilon::Real)
    phistar, Mstar = evolved_luminosity_parameters(g, z)
    return phistar * _effective_density_integral(g, Mstar - Mthr, float(epsilon))
end
background_effective_galaxy_density(g::GalaxyLuminosityFunction, Mthr::Real, z::Real, epsilon::Real) =
    background_effective_galaxy_density(g, Mthr, z; epsilon)
background_effective_galaxy_density(g::GalaxyLuminosityFunction, Mthr::Real, z::Real, r::LogPowerLawAbsMagnitudeRate) =
    background_effective_galaxy_density(g, Mthr, z; epsilon=r.epsilon)
background_effective_galaxy_density(g::GalaxyLuminosityFunction, Mthr::AbstractArray, z::AbstractArray; epsilon::Real) =
    map((m, zi) -> background_effective_galaxy_density(g, m, zi; epsilon), Mthr, z)
background_effective_galaxy_density(g::GalaxyLuminosityFunction, Mthr::AbstractArray, z::Real; epsilon::Real) =
    map(m -> background_effective_galaxy_density(g, m, z; epsilon), Mthr)
background_effective_galaxy_density(g::GalaxyLuminosityFunction, Mthr::Real, z::AbstractArray; epsilon::Real) =
    map(zi -> background_effective_galaxy_density(g, Mthr, zi; epsilon), z)
background_effective_galaxy_density(g::GalaxyLuminosityFunction, Mthr, z, r::LogPowerLawAbsMagnitudeRate) =
    background_effective_galaxy_density(g, Mthr, z; epsilon=r.epsilon)

end

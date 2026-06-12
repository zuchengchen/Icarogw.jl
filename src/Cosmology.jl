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
    sample_comoving_volume

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

end

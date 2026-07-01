module Conversions

using ..Cosmology
using ..SkyMaps
using Distributions
using QuadGK
using Random

export cred_interval,
    chirp_mass,
    mass_ratio,
    f_gw_isco,
    L2M,
    M2L,
    apparent_magnitude,
    absolute_magnitude,
    source_to_detector,
    detector_to_source,
    source2detector,
    detector2source,
    detector_to_source_jacobian,
    detector_to_source_jacobian_q,
    detector_to_source_jacobian_single_mass,
    source_to_detector_jacobian,
    chi_eff_from_spins,
    chi_p_from_spins,
    cartesian_spins_to_chis,
    chi_effective_prior_from_aligned_spins,
    chi_effective_prior_from_isotropic_spins,
    chi_p_prior_from_isotropic_spins,
    chi_p_prior_given_chi_eff_q,
    joint_prior_from_isotropic_spins,
    radec2skymap,
    radec2indeces,
    radec2indices,
    indices2radec,
    MOCMap,
    LigoSkyMap,
    ligo_skymap

"""
    cred_interval(sigma)

Convert a symmetric Gaussian error width in units of `sigma` to enclosed
credible probability.
"""
cred_interval(sigma::Real) = cdf(Normal(), sigma) - cdf(Normal(), -sigma)

const radec2skymap = SkyMaps.radec2skymap
const radec2indeces = SkyMaps.radec2indeces
const radec2indices = SkyMaps.radec2indices
const indices2radec = SkyMaps.indices2radec
const MOCMap = SkyMaps.MOCMap
const LigoSkyMap = SkyMaps.LigoSkyMap
const ligo_skymap = SkyMaps.ligo_skymap

"""
    chirp_mass(m1, m2)

Binary chirp mass in solar masses:
`(m1*m2)^(3/5) / (m1 + m2)^(1/5)`.
"""
chirp_mass(m1::Real, m2::Real) = (m1 * m2)^(3 / 5) / (m1 + m2)^(1 / 5)
chirp_mass(m1::AbstractArray, m2::AbstractArray) = chirp_mass.(m1, m2)

"""
    mass_ratio(m1, m2)

Mass ratio `q = m2 / m1`. The convention assumes `m1 >= m2`, so valid
population-model values are normally in `(0, 1]`.
"""
mass_ratio(m1, m2) = m2 / m1

"""
    f_gw_isco(m1, m2)

Approximate gravitational-wave ISCO frequency in Hz for detector-frame masses
in solar masses. This follows the Python helper `f_GW_ISCO`.
"""
f_gw_isco(m1, m2) = 2 * (2.20 * 10^3 / (m1 + m2))

"""
    L2M(L)

Convert luminosity in watt to absolute bolometric magnitude using the IAU 2015
zero point used by the Python project.
"""
L2M(L) = -2.5 * log10(L) + 71.197425

"""
    M2L(M)

Convert absolute bolometric magnitude to luminosity in watt.
"""
M2L(M) = 3.0128e28 * 10.0^(-0.4 * M)

"""
    apparent_magnitude(M, dl, kcorr=0)

Convert absolute magnitude `M` to apparent magnitude for luminosity distance
`dl` in Mpc and K-correction `kcorr`.
"""
apparent_magnitude(M, dl, kcorr=0) = M + 5log10(dl) + 25 + kcorr

"""
    absolute_magnitude(m, dl, kcorr=0)

Convert apparent magnitude `m` to absolute magnitude for luminosity distance
`dl` in Mpc and K-correction `kcorr`.
"""
absolute_magnitude(m, dl, kcorr=0) = m - 5log10(dl) - 25 - kcorr

"""
    source_to_detector(m1_source, m2_source, z, cosmology)

Convert source-frame component masses to detector-frame masses and luminosity
distance. Returns `(m1_detector, m2_detector, luminosity_distance_Mpc)`.
"""
function source_to_detector(m1_source, m2_source, z, cosmology::AbstractCosmology)
    return m1_source .* (1 .+ z), m2_source .* (1 .+ z), luminosity_distance(cosmology, z)
end
source2detector(args...) = source_to_detector(args...)

"""
    detector_to_source(m1_detector, m2_detector, dl, cosmology)

Convert detector-frame masses and luminosity distance in Mpc to source-frame
masses and redshift. Returns `(m1_source, m2_source, z)`.
"""
function detector_to_source(m1_detector, m2_detector, dl, cosmology::AbstractCosmology)
    z = redshift_at_luminosity_distance(cosmology, dl)
    return m1_detector ./ (1 .+ z), m2_detector ./ (1 .+ z), z
end
detector2source(args...) = detector_to_source(args...)

"""
    detector_to_source_jacobian(z, cosmology)

Jacobian `|d(m1d,m2d,dL) / d(m1s,m2s,z)| = (1+z)^2 ddL/dz`.
"""
detector_to_source_jacobian(z, cosmology::AbstractCosmology) = abs((1 + z)^2 * ddl_dz(cosmology, z))

"""
    detector_to_source_jacobian_q(z, cosmology)

Jacobian `|d(m1d,q,dL) / d(m1s,q,z)| = (1+z) ddL/dz`.
"""
detector_to_source_jacobian_q(z, cosmology::AbstractCosmology) = abs((1 + z) * ddl_dz(cosmology, z))

"""
    detector_to_source_jacobian_single_mass(z, cosmology)

Jacobian `|d(md,dL) / d(ms,z)| = (1+z) ddL/dz`.
"""
detector_to_source_jacobian_single_mass(z, cosmology::AbstractCosmology) =
    detector_to_source_jacobian_q(z, cosmology)

"""
    source_to_detector_jacobian(z, cosmology)

Inverse of `detector_to_source_jacobian`.
"""
source_to_detector_jacobian(z, cosmology::AbstractCosmology) =
    inv(detector_to_source_jacobian(z, cosmology))

"""
    chi_eff_from_spins(chi1, chi2, cos1, cos2, q)

Effective aligned spin `(chi1*cos1 + q*chi2*cos2) / (1 + q)`.
"""
chi_eff_from_spins(chi1, chi2, cos1, cos2, q) = (chi1 * cos1 + q * chi2 * cos2) / (1 + q)
chi_eff_from_spins(chi1::AbstractArray, chi2::AbstractArray, cos1::AbstractArray, cos2::AbstractArray, q) =
    chi_eff_from_spins.(chi1, chi2, cos1, cos2, q)

"""
    chi_p_from_spins(chi1, chi2, cos1, cos2, q)

Approximate precessing-spin parameter following the Python helper.
"""
function chi_p_from_spins(chi1, chi2, cos1, cos2, q)
    s1p = chi1 * sqrt(max(0, 1 - cos1^2))
    s2p = chi2 * sqrt(max(0, 1 - cos2^2))
    return max(s1p, q * (4q + 3) / (4 + 3q) * s2p)
end
chi_p_from_spins(chi1::AbstractArray, chi2::AbstractArray, cos1::AbstractArray, cos2::AbstractArray, q) =
    chi_p_from_spins.(chi1, chi2, cos1, cos2, q)

"""
    chi_effective_prior_from_aligned_spins(q, amax, x)

Conditional prior density `p(chi_eff | q)` for uniform aligned component spins
with maximum magnitude `amax`. This follows the piecewise expression used in
Python `icarogw`.
"""
function chi_effective_prior_from_aligned_spins(q::Real, amax::Real, x::Real)
    abs(x) > amax && return 0.0
    edge = amax * (1 - q) / (1 + q)
    if x > edge
        return (1 + q)^2 * (amax - x) / (4q * amax^2)
    elseif x < -edge
        return (1 + q)^2 * (amax + x) / (4q * amax^2)
    else
        return (1 + q) / (2amax)
    end
end
chi_effective_prior_from_aligned_spins(q, amax, xs::AbstractArray) =
    chi_effective_prior_from_aligned_spins.(q, amax, xs)

_uniform_spin_z_density(amax, y) =
    abs(y) >= amax ? 0.0 : log(amax / max(abs(y), eps(Float64))) / (2amax)

"""
    chi_effective_prior_from_isotropic_spins(q, amax, x)

Conditional prior density `p(chi_eff | q)` for uniform spin magnitudes and
isotropic tilts. The implementation evaluates the convolution of the two
component spin-z distributions by adaptive quadrature, avoiding a Python bridge
or dilogarithm dependency.
"""
function chi_effective_prior_from_isotropic_spins(q::Real, amax::Real, x::Real)
    abs(x) >= amax && return 0.0
    scale1 = 1 / (1 + q)
    scale2 = q / (1 + q)
    f1(y) = _uniform_spin_z_density(amax, y)
    f2(y) = _uniform_spin_z_density(amax, y)
    lo = max(-amax, ((x - scale2 * amax) / scale1))
    hi = min(amax, ((x + scale2 * amax) / scale1))
    lo < hi || return 0.0
    val, _ = quadgk(y -> f1(y) * f2((x - scale1 * y) / scale2) / scale2, lo, hi; rtol=1e-6)
    return max(0.0, val)
end
chi_effective_prior_from_isotropic_spins(q, amax, xs::AbstractArray) =
    chi_effective_prior_from_isotropic_spins.(q, amax, xs)

_chi_perp_density(amax, x) = 0 <= x < amax ? acos(clamp(x / amax, -1, 1)) / amax : 0.0
_scaled_chi_perp_density(amax, scale, x) = scale <= 0 ? 0.0 : _chi_perp_density(amax, x / scale) / scale
_chi_perp_cdf(amax, x) = x <= 0 ? 0.0 : x >= amax ? 1.0 :
    (x * acos(x / amax) - sqrt(max(0.0, amax^2 - x^2)) + amax) / amax
_scaled_chi_perp_cdf(amax, scale, x) = scale <= 0 ? 1.0 : _chi_perp_cdf(amax, x / scale)

"""
    chi_p_prior_from_isotropic_spins(q, amax, x)

Conditional prior density `p(chi_p | q)` for isotropic component spins. It is
computed as the density of the maximum of the primary in-plane spin and the
mass-ratio-scaled secondary in-plane spin.
"""
function chi_p_prior_from_isotropic_spins(q::Real, amax::Real, x::Real)
    scale = q * (4q + 3) / (4 + 3q)
    f1 = _chi_perp_density(amax, x)
    F1 = _chi_perp_cdf(amax, x)
    f2 = _scaled_chi_perp_density(amax, scale, x)
    F2 = _scaled_chi_perp_cdf(amax, scale, x)
    return f1 * F2 + f2 * F1
end
chi_p_prior_from_isotropic_spins(q, amax, xs::AbstractArray) =
    chi_p_prior_from_isotropic_spins.(q, amax, xs)

function _trapz(xs::AbstractVector, ys::AbstractVector)
    length(xs) == length(ys) || throw(ArgumentError("trapz inputs must have matching lengths"))
    length(xs) < 2 && return 0.0
    total = 0.0
    @inbounds for i in 1:(length(xs) - 1)
        total += 0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i])
    end
    return total
end

function _interp_linear(x::Real, xs::AbstractVector, ys::AbstractVector)
    x <= first(xs) && return first(ys)
    x >= last(xs) && return last(ys)
    i = searchsortedlast(xs, x)
    x0, x1 = xs[i], xs[i + 1]
    y0, y1 = ys[i], ys[i + 1]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
end

function _kde_bandwidth_factor(weights::AbstractVector, bw_method)
    sw = sum(weights)
    sw > 0 || throw(ArgumentError("KDE weights must have positive sum"))
    normalized = weights ./ sw
    neff = inv(sum(abs2, normalized))
    if bw_method === :scott || bw_method == "scott"
        return neff^(-1 / 5)
    elseif bw_method === :silverman || bw_method == "silverman"
        return (0.75 * neff)^(-1 / 5)
    elseif bw_method isa Real
        bw_method > 0 || throw(ArgumentError("numeric bw_method must be positive"))
        return float(bw_method)
    else
        throw(ArgumentError("bw_method must be :scott, :silverman, or a positive number"))
    end
end

function _weighted_kde_1d(samples::AbstractVector, weights::AbstractVector, grid::AbstractVector; bw_method=:scott)
    length(samples) == length(weights) || throw(ArgumentError("KDE samples and weights must have matching lengths"))
    sw = sum(weights)
    sw > 0 || throw(ArgumentError("KDE weights must have positive sum"))
    w = weights ./ sw
    mu = sum(w .* samples)
    variance = sum(w .* (samples .- mu) .^ 2)
    bandwidth = sqrt(max(variance, eps(Float64))) * _kde_bandwidth_factor(weights, bw_method)
    inv_norm = inv(bandwidth * sqrt(2π))
    values = zeros(Float64, length(grid))
    @inbounds for (i, x) in pairs(grid)
        acc = 0.0
        for j in eachindex(samples)
            z = (x - samples[j]) / bandwidth
            acc += w[j] * exp(-0.5 * z^2)
        end
        values[i] = inv_norm * acc
    end
    return values
end

function _conditional_max_chi_p(q::Real, amax::Real, xeff::Real)
    if (1 + q) * abs(xeff) / q < amax
        return float(amax)
    end
    arg = amax^2 - ((1 + q) * abs(xeff) - q)^2
    return arg > 0 ? sqrt(arg) : 0.0
end

"""
    chi_p_prior_given_chi_eff_q([rng], q, amax, xeff, xp; ndraws=10000, bw_method=:scott)

Monte Carlo estimate of the conditional isotropic-spin prior
`p(chi_p | chi_eff, q)`, matching the Python helper's rejection sampling and
weighted one-dimensional KDE workflow. Pass an explicit RNG for reproducible
fixtures.
"""
function chi_p_prior_given_chi_eff_q(rng::AbstractRNG, q::Real, amax::Real, xeff::Real, xp::Real;
    ndraws::Integer=10000, bw_method=:scott)
    0 < q <= 1 || throw(ArgumentError("q must lie in (0, 1]"))
    amax > 0 || throw(ArgumentError("amax must be positive"))
    ndraws > 1 || throw(ArgumentError("ndraws must be greater than 1"))
    abs(xeff) < amax || return 0.0
    max_chi_p = _conditional_max_chi_p(q, amax, xeff)
    max_chi_p > 0 || return 0.0

    a1 = rand(rng, ndraws) .* amax
    a2 = rand(rng, ndraws) .* amax
    cos2 = 2 .* rand(rng, ndraws) .- 1
    cos1 = similar(a1)
    invalid = trues(ndraws)
    while any(invalid)
        @inbounds for i in eachindex(a1)
            cos1[i] = (xeff * (1 + q) - q * a2[i] * cos2[i]) / a1[i]
            invalid[i] = cos1[i] < -1 || cos1[i] > 1 || !isfinite(cos1[i])
        end
        nbad = count(invalid)
        nbad == 0 && break
        a1[invalid] .= rand(rng, nbad) .* amax
        a2[invalid] .= rand(rng, nbad) .* amax
        cos2[invalid] .= 2 .* rand(rng, nbad) .- 1
    end

    xp_draws = similar(a1)
    @inbounds for i in eachindex(a1)
        xp_draws[i] = chi_p_from_spins(a1[i], a2[i], cos1[i], cos2[i], q)
    end
    jacobian_weights = (1 + q) ./ a1
    reference_grid = collect(range(0.05 * max_chi_p, 0.95 * max_chi_p; length=50))
    reference_vals = _weighted_kde_1d(xp_draws, jacobian_weights, reference_grid; bw_method)
    pushfirst!(reference_grid, 0.0)
    pushfirst!(reference_vals, 0.0)
    push!(reference_grid, max_chi_p)
    push!(reference_vals, 0.0)
    norm_constant = _trapz(reference_grid, reference_vals)
    norm_constant > 0 && isfinite(norm_constant) || return 0.0
    return _interp_linear(xp, reference_grid, reference_vals ./ norm_constant)
end
chi_p_prior_given_chi_eff_q(q::Real, amax::Real, xeff::Real, xp::Real; kwargs...) =
    chi_p_prior_given_chi_eff_q(Random.default_rng(), q, amax, xeff, xp; kwargs...)

_q_value(q::Real, _) = q
_q_value(q::AbstractArray, i) = q[i]

"""
    joint_prior_from_isotropic_spins([rng], q, amax, xeffs, xps; ndraws=10000, bw_method=:scott)

Estimate the joint isotropic-spin prior
`p(chi_eff, chi_p | q) = p(chi_eff | q) p(chi_p | chi_eff, q)` for matching
arrays of effective spin samples. `q` may be scalar or array-valued. Pass an
explicit RNG for reproducible Monte Carlo/KDE results.
"""
function joint_prior_from_isotropic_spins(rng::AbstractRNG, q, amax::Real, xeffs::AbstractArray, xps::AbstractArray;
    ndraws::Integer=10000, bw_method=:scott)
    axes(xeffs) == axes(xps) || throw(ArgumentError("xeffs and xps must have matching axes"))
    if q isa AbstractArray
        axes(q) == axes(xeffs) || throw(ArgumentError("array q must have the same axes as xeffs"))
    end
    out = similar(Float64.(xeffs))
    @inbounds for i in eachindex(xeffs, xps)
        qi = _q_value(q, i)
        p_eff = chi_effective_prior_from_isotropic_spins(qi, amax, xeffs[i])
        p_cond = chi_p_prior_given_chi_eff_q(rng, qi, amax, xeffs[i], xps[i]; ndraws, bw_method)
        out[i] = p_eff * p_cond
    end
    return out
end
joint_prior_from_isotropic_spins(q, amax::Real, xeffs::AbstractArray, xps::AbstractArray; kwargs...) =
    joint_prior_from_isotropic_spins(Random.default_rng(), q, amax, xeffs, xps; kwargs...)

"""
    cartesian_spins_to_chis(s1x, s1y, s1z, s2x, s2y, s2z, q)

Convert Cartesian dimensionless spin components to `(chi_eff, chi_p)`.
"""
function cartesian_spins_to_chis(s1x, s1y, s1z, s2x, s2y, s2z, q)
    chi1 = sqrt(s1x^2 + s1y^2 + s1z^2)
    chi2 = sqrt(s2x^2 + s2y^2 + s2z^2)
    cos1 = chi1 == 0 ? 1.0 : s1z / chi1
    cos2 = chi2 == 0 ? 1.0 : s2z / chi2
    return chi_eff_from_spins(chi1, chi2, cos1, cos2, q),
        chi_p_from_spins(chi1, chi2, cos1, cos2, q)
end

end

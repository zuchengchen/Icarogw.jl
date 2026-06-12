module Priors

using Distributions
using QuadGK
using Random
using SpecialFunctions
using Statistics
using StatsBase
using ..Utils: logaddexp

export AbstractPrior,
    PowerLaw,
    BetaDistribution,
    TruncatedBetaDistribution,
    TruncatedGaussian,
    PowerLawGaussian,
    BrokenPowerLaw,
    PowerLawTwoGaussians,
    BrokenPowerLawMultiPeak,
    BrokenPowerLawTripleMultiPeak,
    ConditionalMassDistribution,
    PairedMassDistribution,
    PiecewiseConstant2D,
    LowpassSmoothedProb,
    SmoothedPlusDipProb,
    PowerLawStationary,
    PowerLawLinear,
    GaussianStationary,
    GaussianLinear,
    MixtureMassPrior,
    DefaultSpinPrior,
    GaussianSpinPrior,
    logpdf,
    pdf,
    cdf,
    rand_prior,
    ParameterSpec,
    ParameterSchema,
    parameter_schema,
    prior_transform,
    pack,
    unpack,
    betadistro_muvar2ab,
    betadistro_ab2muvar

abstract type AbstractPrior end

_inrange(x, lo, hi) = lo <= x <= hi
_bounds_logpdf(x, lo, hi, value) = _inrange(x, lo, hi) ? value : -Inf

logpdf(p::AbstractPrior, x::AbstractArray) = map(v -> logpdf(p, v), x)
pdf(p::AbstractPrior, x) = exp.(logpdf(p, x))
cdf(p::AbstractPrior, x::AbstractArray) = map(v -> cdf(p, v), x)
logpdf(p::AbstractPrior, x::AbstractArray, z::AbstractArray) = map(logpdf, Ref(p), x, z)
pdf(p::AbstractPrior, x, z) = exp.(logpdf(p, x, z))

"""
    PowerLaw(min, max, alpha)

Normalized one-dimensional power-law probability density
`p(x) ∝ x^alpha` on `[min, max]`. This stores the exponent directly; Python
mass-prior wrappers often pass `-alpha`.
"""
struct PowerLaw <: AbstractPrior
    min::Float64
    max::Float64
    alpha::Float64
    norm::Float64
    function PowerLaw(min::Real, max::Real, alpha::Real)
        min < max || throw(ArgumentError("PowerLaw requires min < max"))
        min >= 0 || throw(ArgumentError("PowerLaw minimum must be non-negative"))
        a = Float64(alpha)
        norm = a == -1 ? log(max / min) : (max^(a + 1) - min^(a + 1)) / (a + 1)
        norm > 0 || throw(ArgumentError("invalid PowerLaw normalization"))
        return new(float(min), float(max), a, norm)
    end
end
logpdf(p::PowerLaw, x::Real) = _bounds_logpdf(x, p.min, p.max, p.alpha * log(x) - log(p.norm))
function cdf(p::PowerLaw, x::Real)
    x <= p.min && return 0.0
    x >= p.max && return 1.0
    return p.alpha == -1 ? log(x / p.min) / p.norm :
        ((x^(p.alpha + 1) - p.min^(p.alpha + 1)) / (p.alpha + 1)) / p.norm
end
function rand_prior(rng::AbstractRNG, p::PowerLaw, n::Integer)
    u = rand(rng, n)
    if p.alpha == -1
        return p.min .* exp.(u .* p.norm)
    end
    a1 = p.alpha + 1
    return (p.min^a1 .+ u .* (p.max^a1 - p.min^a1)).^(1 / a1)
end
rand_prior(p::PowerLaw, n::Integer) = rand_prior(Random.default_rng(), p, n)

"""
    BetaDistribution(alpha, beta)

Beta probability density on `[0, 1]`.
"""
struct BetaDistribution <: AbstractPrior
    alpha::Float64
    beta::Float64
    dist::Distributions.Beta{Float64}
end
function BetaDistribution(alpha::Real, beta::Real)
    alpha > 0 && beta > 0 || throw(ArgumentError("Beta parameters must be positive"))
    return BetaDistribution(float(alpha), float(beta), Distributions.Beta(float(alpha), float(beta)))
end
logpdf(p::BetaDistribution, x::Real) = _inrange(x, 0, 1) ? Distributions.logpdf(p.dist, x) : -Inf
cdf(p::BetaDistribution, x::Real) = Distributions.cdf(p.dist, clamp(float(x), 0, 1))
rand_prior(rng::AbstractRNG, p::BetaDistribution, n::Integer) = rand(rng, p.dist, n)

"""
    TruncatedBetaDistribution(alpha, beta, maximum)

Beta probability density truncated to `[0, maximum]`.
"""
struct TruncatedBetaDistribution <: AbstractPrior
    alpha::Float64
    beta::Float64
    maximum::Float64
    base::BetaDistribution
    norm::Float64
end
function TruncatedBetaDistribution(alpha::Real, beta::Real, maximum::Real)
    0 < maximum <= 1 || throw(ArgumentError("maximum must lie in (0, 1]"))
    base = BetaDistribution(alpha, beta)
    return TruncatedBetaDistribution(float(alpha), float(beta), float(maximum), base, cdf(base, maximum))
end
logpdf(p::TruncatedBetaDistribution, x::Real) = _inrange(x, 0, p.maximum) ? logpdf(p.base, x) - log(p.norm) : -Inf
cdf(p::TruncatedBetaDistribution, x::Real) = x <= 0 ? 0.0 : x >= p.maximum ? 1.0 : cdf(p.base, x) / p.norm
function rand_prior(rng::AbstractRNG, p::TruncatedBetaDistribution, n::Integer)
    out = Vector{Float64}(undef, n)
    i = 1
    while i <= n
        x = rand(rng, p.base.dist)
        if x <= p.maximum
            out[i] = x
            i += 1
        end
    end
    return out
end

"""
    TruncatedGaussian(mean, sigma, min, max)

Normal probability density truncated to `[min, max]`.
"""
struct TruncatedGaussian <: AbstractPrior
    mean::Float64
    sigma::Float64
    min::Float64
    max::Float64
    base::Normal{Float64}
    norm::Float64
end
function TruncatedGaussian(mean::Real, sigma::Real, min::Real, max::Real)
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    min < max || throw(ArgumentError("TruncatedGaussian requires min < max"))
    base = Normal(float(mean), float(sigma))
    norm = Distributions.cdf(base, max) - Distributions.cdf(base, min)
    return TruncatedGaussian(float(mean), float(sigma), float(min), float(max), base, norm)
end
logpdf(p::TruncatedGaussian, x::Real) = _inrange(x, p.min, p.max) ? Distributions.logpdf(p.base, x) - log(p.norm) : -Inf
cdf(p::TruncatedGaussian, x::Real) =
    x <= p.min ? 0.0 : x >= p.max ? 1.0 : (Distributions.cdf(p.base, x) - Distributions.cdf(p.base, p.min)) / p.norm
function rand_prior(rng::AbstractRNG, p::TruncatedGaussian, n::Integer)
    out = Vector{Float64}(undef, n)
    i = 1
    while i <= n
        x = rand(rng, p.base)
        if p.min <= x <= p.max
            out[i] = x
            i += 1
        end
    end
    return out
end

"""
    PowerLawGaussian(minpl, maxpl, alpha, lambda_peak, mean, sigma, ming, maxg)

Mixture of a normalized `PowerLaw` and a `TruncatedGaussian`.
"""
struct PowerLawGaussian <: AbstractPrior
    pl::PowerLaw
    gauss::TruncatedGaussian
    lambda::Float64
    min::Float64
    max::Float64
end
function PowerLawGaussian(minpl, maxpl, alpha, lambda_peak, mean, sigma, ming, maxg)
    0 <= lambda_peak <= 1 || throw(ArgumentError("lambda_peak must be in [0, 1]"))
    pl = PowerLaw(minpl, maxpl, alpha)
    g = TruncatedGaussian(mean, sigma, ming, maxg)
    return PowerLawGaussian(pl, g, float(lambda_peak), min(float(minpl), float(ming)), max(float(maxpl), float(maxg)))
end
function logpdf(p::PowerLawGaussian, x::Real)
    a = p.lambda == 1 ? -Inf : log1p(-p.lambda) + logpdf(p.pl, x)
    b = p.lambda == 0 ? -Inf : log(p.lambda) + logpdf(p.gauss, x)
    return logaddexp(a, b)
end
cdf(p::PowerLawGaussian, x::Real) = (1 - p.lambda) * cdf(p.pl, x) + p.lambda * cdf(p.gauss, x)

"""
    BrokenPowerLaw(min, max, alpha1, alpha2, b)

Continuous broken power law on `[min, max]`; `b` is the fractional break
location, matching the Python wrapper.
"""
struct BrokenPowerLaw <: AbstractPrior
    pl1::PowerLaw
    pl2::PowerLaw
    break_point::Float64
    ratio::Float64
    norm::Float64
    min::Float64
    max::Float64
end
function BrokenPowerLaw(min, max, alpha1, alpha2, b)
    0 < b < 1 || throw(ArgumentError("b must lie in (0, 1)"))
    bp = min + b * (max - min)
    pl1 = PowerLaw(min, bp, alpha1)
    pl2 = PowerLaw(bp, max, alpha2)
    ratio = exp(logpdf(pl1, bp) - logpdf(pl2, bp))
    return BrokenPowerLaw(pl1, pl2, bp, ratio, 1 + ratio, float(min), float(max))
end
function logpdf(p::BrokenPowerLaw, x::Real)
    _inrange(x, p.min, p.max) || return -Inf
    return x <= p.break_point ? logpdf(p.pl1, x) - log(p.norm) :
        logpdf(p.pl2, x) + log(p.ratio) - log(p.norm)
end
function cdf(p::BrokenPowerLaw, x::Real)
    x <= p.min && return 0.0
    x >= p.max && return 1.0
    return (cdf(p.pl1, x) + p.ratio * cdf(p.pl2, x)) / p.norm
end

"""
    PowerLawTwoGaussians(...)

Mixture of one power law and two truncated Gaussian peaks.
"""
struct PowerLawTwoGaussians <: AbstractPrior
    pl::PowerLaw
    glow::TruncatedGaussian
    ghigh::TruncatedGaussian
    lambda::Float64
    lambda_low::Float64
    min::Float64
    max::Float64
end
function PowerLawTwoGaussians(minpl, maxpl, alpha, lambda_g, lambda_low, mean_low, sigma_low, min_low, max_low, mean_high, sigma_high, min_high, max_high)
    0 <= lambda_g <= 1 && 0 <= lambda_low <= 1 || throw(ArgumentError("mixture weights must be in [0, 1]"))
    pl = PowerLaw(minpl, maxpl, alpha)
    gl = TruncatedGaussian(mean_low, sigma_low, min_low, max_low)
    gh = TruncatedGaussian(mean_high, sigma_high, min_high, max_high)
    return PowerLawTwoGaussians(pl, gl, gh, float(lambda_g), float(lambda_low),
        minimum((float(minpl), float(min_low), float(min_high))), maximum((float(maxpl), float(max_low), float(max_high))))
end
function logpdf(p::PowerLawTwoGaussians, x::Real)
    a = p.lambda == 1 ? -Inf : log1p(-p.lambda) + logpdf(p.pl, x)
    b = p.lambda == 0 || p.lambda_low == 0 ? -Inf : log(p.lambda) + log(p.lambda_low) + logpdf(p.glow, x)
    cval = p.lambda == 0 || p.lambda_low == 1 ? -Inf : log(p.lambda) + log1p(-p.lambda_low) + logpdf(p.ghigh, x)
    return logaddexp(logaddexp(a, b), cval)
end
cdf(p::PowerLawTwoGaussians, x::Real) =
    (1 - p.lambda) * cdf(p.pl, x) + p.lambda * (p.lambda_low * cdf(p.glow, x) + (1 - p.lambda_low) * cdf(p.ghigh, x))

"""
    BrokenPowerLawMultiPeak(...)

Mixture of a broken power law and two Gaussian peaks.
"""
struct BrokenPowerLawMultiPeak <: AbstractPrior
    bpl::BrokenPowerLaw
    glow::TruncatedGaussian
    ghigh::TruncatedGaussian
    lambda::Float64
    lambda_low::Float64
end
function BrokenPowerLawMultiPeak(minpl, maxpl, alpha1, alpha2, b, lambda_g, lambda_low, mean_low, sigma_low, min_low, max_low, mean_high, sigma_high, min_high, max_high)
    return BrokenPowerLawMultiPeak(
        BrokenPowerLaw(minpl, maxpl, alpha1, alpha2, b),
        TruncatedGaussian(mean_low, sigma_low, min_low, max_low),
        TruncatedGaussian(mean_high, sigma_high, min_high, max_high),
        float(lambda_g),
        float(lambda_low),
    )
end
function logpdf(p::BrokenPowerLawMultiPeak, x::Real)
    a = p.lambda == 1 ? -Inf : log1p(-p.lambda) + logpdf(p.bpl, x)
    b = p.lambda == 0 || p.lambda_low == 0 ? -Inf : log(p.lambda) + log(p.lambda_low) + logpdf(p.glow, x)
    cval = p.lambda == 0 || p.lambda_low == 1 ? -Inf : log(p.lambda) + log1p(-p.lambda_low) + logpdf(p.ghigh, x)
    return logaddexp(logaddexp(a, b), cval)
end
cdf(p::BrokenPowerLawMultiPeak, x::Real) =
    (1 - p.lambda) * cdf(p.bpl, x) + p.lambda * (p.lambda_low * cdf(p.glow, x) + (1 - p.lambda_low) * cdf(p.ghigh, x))

"""
    BrokenPowerLawTripleMultiPeak(...)

Mixture of a broken power law and three truncated Gaussian peaks.
"""
struct BrokenPowerLawTripleMultiPeak <: AbstractPrior
    bpl::BrokenPowerLaw
    g1::TruncatedGaussian
    g2::TruncatedGaussian
    g3::TruncatedGaussian
    lambda::Float64
    lambda1::Float64
    lambda2::Float64
end
function BrokenPowerLawTripleMultiPeak(minpl, maxpl, alpha1, alpha2, b, lambda_g, lambda1, lambda2,
    mean1, sigma1, min1, max1, mean2, sigma2, min2, max2, mean3, sigma3, min3, max3)
    return BrokenPowerLawTripleMultiPeak(BrokenPowerLaw(minpl, maxpl, alpha1, alpha2, b),
        TruncatedGaussian(mean1, sigma1, min1, max1),
        TruncatedGaussian(mean2, sigma2, min2, max2),
        TruncatedGaussian(mean3, sigma3, min3, max3),
        float(lambda_g), float(lambda1), float(lambda2))
end
function logpdf(p::BrokenPowerLawTripleMultiPeak, x::Real)
    a = p.lambda == 1 ? -Inf : log1p(-p.lambda) + logpdf(p.bpl, x)
    g1 = p.lambda == 0 || p.lambda1 == 0 ? -Inf : log(p.lambda) + log(p.lambda1) + logpdf(p.g1, x)
    g2 = p.lambda == 0 || p.lambda1 == 1 || p.lambda2 == 0 ? -Inf :
        log(p.lambda) + log1p(-p.lambda1) + log(p.lambda2) + logpdf(p.g2, x)
    g3 = p.lambda == 0 || p.lambda1 == 1 || p.lambda2 == 1 ? -Inf :
        log(p.lambda) + log1p(-p.lambda1) + log1p(-p.lambda2) + logpdf(p.g3, x)
    return logaddexp(logaddexp(logaddexp(a, g1), g2), g3)
end
cdf(p::BrokenPowerLawTripleMultiPeak, x::Real) =
    (1 - p.lambda) * cdf(p.bpl, x) +
    p.lambda * (p.lambda1 * cdf(p.g1, x) + (1 - p.lambda1) * (p.lambda2 * cdf(p.g2, x) + (1 - p.lambda2) * cdf(p.g3, x)))

"""
    ConditionalMassDistribution(primary, secondary)

Two-dimensional distribution `p(m1, m2) = p1(m1) p2(m2) / CDF2(m1)` with
support `m2 <= m1`.
"""
struct ConditionalMassDistribution{P1<:AbstractPrior,P2<:AbstractPrior} <: AbstractPrior
    primary::P1
    secondary::P2
end
function logpdf(p::ConditionalMassDistribution, m1::Real, m2::Real)
    m2 <= m1 || return -Inf
    denom = cdf(p.secondary, m1)
    denom > 0 || return -Inf
    return logpdf(p.primary, m1) + logpdf(p.secondary, m2) - log(denom)
end
pdf(p::ConditionalMassDistribution, m1::Real, m2::Real) = exp(logpdf(p, m1, m2))
function rand_prior(rng::AbstractRNG, p::ConditionalMassDistribution, n::Integer)
    m1 = rand_prior(rng, p.primary, n)
    m2 = Vector{Float64}(undef, n)
    for i in eachindex(m1)
        target = rand(rng) * cdf(p.secondary, m1[i])
        lo = getfield(p.secondary, :min)
        hi = min(m1[i], getfield(p.secondary, :max))
        for _ in 1:80
            mid = (lo + hi) / 2
            if cdf(p.secondary, mid) < target
                lo = mid
            else
                hi = mid
            end
        end
        m2[i] = (lo + hi) / 2
    end
    return m1, m2
end

"""
    PairedMassDistribution(base; beta=0)

Two-dimensional distribution proportional to `base(m1) base(m2) q^beta` on
`m2 <= m1`, where `q = m2/m1`.
"""
struct PairedMassDistribution{P<:AbstractPrior} <: AbstractPrior
    base::P
    beta::Float64
    norm::Float64
end
function PairedMassDistribution(base::AbstractPrior; beta::Real=0)
    lo = getfield(base, :min)
    hi = getfield(base, :max)
    inner(m1) = quadgk(m2 -> pdf(base, m2) * (m2 / m1)^beta, lo, m1; rtol=1e-6)[1]
    norm = quadgk(m1 -> pdf(base, m1) * inner(m1), lo, hi; rtol=1e-5)[1]
    return PairedMassDistribution(base, float(beta), norm)
end
function logpdf(p::PairedMassDistribution, m1::Real, m2::Real)
    m2 <= m1 || return -Inf
    q = m2 / m1
    q > 0 || return -Inf
    return logpdf(p.base, m1) + logpdf(p.base, m2) + p.beta * log(q) - log(p.norm)
end
pdf(p::PairedMassDistribution, m1::Real, m2::Real) = exp(logpdf(p, m1, m2))

"""
    PiecewiseConstant2D(min, max, weights)

Normalized triangular checkerboard distribution for `(m1, m2)` with `m2 <= m1`.
Weights are ordered as in the Python implementation: `(0,0), (1,0), (2,0),
(1,1), (2,1), (2,2), ...`.
"""
struct PiecewiseConstant2D <: AbstractPrior
    min::Float64
    max::Float64
    weights::Vector{Float64}
    n1d::Int
    dx::Float64
    normalized::Vector{Float64}
end
function PiecewiseConstant2D(min::Real, max::Real, weights::AbstractVector{<:Real})
    n = Int(round(-0.5 + sqrt(0.25 + 2length(weights))))
    n * (n + 1) ÷ 2 == length(weights) || throw(ArgumentError("weights length must be triangular"))
    dx = (max - min) / n
    area_weights = 0.0
    k = 1
    for j in 0:(n - 1), i in j:(n - 1)
        area_weights += weights[k] * (i == j ? 0.5 : 1.0)
        k += 1
    end
    norm = 1 / (area_weights * dx^2)
    return PiecewiseConstant2D(float(min), float(max), Float64.(weights), n, dx, norm .* Float64.(weights))
end
function _piece_index(p::PiecewiseConstant2D, m1, m2)
    i = clamp(fld(Int(floor((m1 - p.min) / p.dx)), 1) + 1, 1, p.n1d)
    j = clamp(fld(Int(floor((m2 - p.min) / p.dx)), 1) + 1, 1, p.n1d)
    k = 1
    for jj in 1:p.n1d, ii in jj:p.n1d
        ii == i && jj == j && return k
        k += 1
    end
    return 0
end
function pdf(p::PiecewiseConstant2D, m1::Real, m2::Real)
    p.min <= m2 <= m1 <= p.max || return 0.0
    k = _piece_index(p, m1, m2)
    return k == 0 ? 0.0 : p.normalized[k]
end
logpdf(p::PiecewiseConstant2D, m1::Real, m2::Real) = (v = pdf(p, m1, m2); v > 0 ? log(v) : -Inf)

_sigmoid_window(x, edge, delta, highpass::Bool) = delta == 0 ? 1.0 :
    highpass ? (x <= edge ? 0.0 : x >= edge + delta ? 1.0 :
        1 / (1 + exp(delta / (x - edge) + delta / (x - edge - delta)))) :
    (x >= edge ? 0.0 : x <= edge - delta ? 1.0 :
        1 / (1 + exp(delta / (edge - x) + delta / (edge - x - delta))))
_notch(x, left, dleft, right, dright, depth) =
    1 - depth * _sigmoid_window(x, left, dleft, true) * _sigmoid_window(x, right, dright, false)

"""
    LowpassSmoothedProb(origin, delta_m)

Low-mass smoothed version of a one-dimensional prior. The name follows the
Python class even though the implemented window is a high-pass turn-on.
"""
struct LowpassSmoothedProb{P<:AbstractPrior} <: AbstractPrior
    origin::P
    delta::Float64
    min::Float64
    max::Float64
    norm::Float64
end
function LowpassSmoothedProb(origin::AbstractPrior, delta::Real)
    lo = getfield(origin, :min)
    hi = getfield(origin, :max)
    d = float(delta)
    norm = quadgk(x -> pdf(origin, x) * _sigmoid_window(x, lo, d, true), lo, hi; rtol=1e-6)[1]
    return LowpassSmoothedProb(origin, d, lo, hi, norm)
end
function logpdf(p::LowpassSmoothedProb, x::Real)
    _inrange(x, p.min, p.max) || return -Inf
    w = _sigmoid_window(x, p.min, p.delta, true)
    w > 0 || return -Inf
    return logpdf(p.origin, x) + log(w) - log(p.norm)
end
cdf(p::LowpassSmoothedProb, x::Real) =
    x <= p.min ? 0.0 : x >= p.max ? 1.0 : quadgk(t -> pdf(p, t), p.min, x; rtol=1e-5)[1]

"""
    SmoothedPlusDipProb(origin, bottomsmooth, topsmooth, leftdip, rightdip, leftdipsmooth, rightdipsmooth, deep)

Smoothed one-dimensional prior with a notch-like dip between `leftdip` and
`rightdip`.
"""
struct SmoothedPlusDipProb{P<:AbstractPrior} <: AbstractPrior
    origin::P
    bottomsmooth::Float64
    topsmooth::Float64
    leftdip::Float64
    rightdip::Float64
    leftdipsmooth::Float64
    rightdipsmooth::Float64
    deep::Float64
    min::Float64
    max::Float64
    norm::Float64
end
function SmoothedPlusDipProb(origin::AbstractPrior, bottomsmooth, topsmooth, leftdip, rightdip, leftdipsmooth, rightdipsmooth, deep)
    lo = getfield(origin, :min)
    hi = getfield(origin, :max)
    f(x) = pdf(origin, x) * _sigmoid_window(x, lo, bottomsmooth, true) *
        _sigmoid_window(x, hi, topsmooth, false) * _notch(x, leftdip, leftdipsmooth, rightdip, rightdipsmooth, deep)
    norm = quadgk(f, lo, hi; rtol=1e-6)[1]
    return SmoothedPlusDipProb(origin, float(bottomsmooth), float(topsmooth), float(leftdip), float(rightdip),
        float(leftdipsmooth), float(rightdipsmooth), float(deep), lo, hi, norm)
end
function logpdf(p::SmoothedPlusDipProb, x::Real)
    _inrange(x, p.min, p.max) || return -Inf
    w = _sigmoid_window(x, p.min, p.bottomsmooth, true) * _sigmoid_window(x, p.max, p.topsmooth, false) *
        _notch(x, p.leftdip, p.leftdipsmooth, p.rightdip, p.rightdipsmooth, p.deep)
    w > 0 || return -Inf
    return logpdf(p.origin, x) + log(w) - log(p.norm)
end
cdf(p::SmoothedPlusDipProb, x::Real) =
    x <= p.min ? 0.0 : x >= p.max ? 1.0 : quadgk(t -> pdf(p, t), p.min, x; rtol=1e-5)[1]

"""
    PowerLawStationary(alpha, mmin, mmax)

Population mass model with density proportional to `m^(-alpha)` on
`[mmin, mmax]`. This mirrors Python `PowerLawStationary`.
"""
struct PowerLawStationary <: AbstractPrior
    alpha::Float64
    min::Float64
    max::Float64
    prior::PowerLaw
end
PowerLawStationary(alpha, mmin, mmax) =
    PowerLawStationary(float(alpha), float(mmin), float(mmax), PowerLaw(mmin, mmax, -alpha))
logpdf(p::PowerLawStationary, m::Real) = logpdf(p.prior, m)
cdf(p::PowerLawStationary, m::Real) = cdf(p.prior, m)

"""
    PowerLawLinear(alpha_z0, alpha_z1, mmin_z0, mmin_z1, mmax_z0, mmax_z1)

Redshift-linear power-law mass model. At redshift `z`, the exponent is
`-(alpha_z0 + alpha_z1*z)` and support is
`[mmin_z0 + mmin_z1*z, mmax_z0 + mmax_z1*z]`.
"""
struct PowerLawLinear <: AbstractPrior
    alpha_z0::Float64
    alpha_z1::Float64
    mmin_z0::Float64
    mmin_z1::Float64
    mmax_z0::Float64
    mmax_z1::Float64
    min::Float64
    max::Float64
end
function PowerLawLinear(alpha_z0, alpha_z1, mmin_z0, mmin_z1, mmax_z0, mmax_z1)
    return PowerLawLinear(float(alpha_z0), float(alpha_z1), float(mmin_z0), float(mmin_z1),
        float(mmax_z0), float(mmax_z1), -Inf, Inf)
end
function logpdf(p::PowerLawLinear, m::Real, z::Real)
    lo = p.mmin_z0 + p.mmin_z1 * z
    hi = p.mmax_z0 + p.mmax_z1 * z
    lo < hi || return -Inf
    return logpdf(PowerLaw(lo, hi, -(p.alpha_z0 + p.alpha_z1 * z)), m)
end

"""
    GaussianStationary(mu, sigma, mmin)

Gaussian mass model truncated below `mmin` and unbounded above.
"""
struct GaussianStationary <: AbstractPrior
    mu::Float64
    sigma::Float64
    min::Float64
    max::Float64
    base::Normal{Float64}
    norm::Float64
end
function GaussianStationary(mu, sigma, mmin)
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    base = Normal(float(mu), float(sigma))
    norm = 1 - Distributions.cdf(base, float(mmin))
    return GaussianStationary(float(mu), float(sigma), float(mmin), Inf, base, norm)
end
logpdf(p::GaussianStationary, m::Real) =
    m < p.min ? -Inf : Distributions.logpdf(p.base, m) - log(p.norm)
cdf(p::GaussianStationary, m::Real) =
    m <= p.min ? 0.0 : (Distributions.cdf(p.base, m) - Distributions.cdf(p.base, p.min)) / p.norm

"""
    GaussianLinear(mu_z0, mu_z1, sigma_z0, sigma_z1, mmin)

Redshift-linear Gaussian mass model truncated below `mmin`.
"""
struct GaussianLinear <: AbstractPrior
    mu_z0::Float64
    mu_z1::Float64
    sigma_z0::Float64
    sigma_z1::Float64
    min::Float64
    max::Float64
end
function GaussianLinear(mu_z0, mu_z1, sigma_z0, sigma_z1, mmin)
    return GaussianLinear(float(mu_z0), float(mu_z1), float(sigma_z0), float(sigma_z1), float(mmin), Inf)
end
function logpdf(p::GaussianLinear, m::Real, z::Real)
    sigma = p.sigma_z0 + p.sigma_z1 * z
    sigma > 0 || return -Inf
    return logpdf(GaussianStationary(p.mu_z0 + p.mu_z1 * z, sigma, p.min), m)
end

"""
    MixtureMassPrior(components, weights)

Mixture of one-dimensional mass priors. Components may be stationary
(`logpdf(component, m)`) or redshift dependent (`logpdf(component, m, z)`).
Weights are normalized at construction time.
"""
struct MixtureMassPrior{C<:Tuple} <: AbstractPrior
    components::C
    weights::Vector{Float64}
    min::Float64
    max::Float64
end
function MixtureMassPrior(components::Tuple, weights::AbstractVector{<:Real})
    length(components) == length(weights) || throw(ArgumentError("components and weights length mismatch"))
    all(>=(0), weights) || throw(ArgumentError("mixture weights must be non-negative"))
    total = sum(weights)
    total > 0 || throw(ArgumentError("at least one mixture weight must be positive"))
    mins = [hasfield(typeof(c), :min) ? getfield(c, :min) : -Inf for c in components]
    maxs = [hasfield(typeof(c), :max) ? getfield(c, :max) : Inf for c in components]
    return MixtureMassPrior(components, Float64.(weights) ./ total, minimum(mins), maximum(maxs))
end
MixtureMassPrior(components::AbstractVector, weights::AbstractVector{<:Real}) =
    MixtureMassPrior(Tuple(components), weights)
_component_logpdf(c, m) = logpdf(c, m)
_component_logpdf(c, m, z) = applicable(logpdf, c, m, z) ? logpdf(c, m, z) : logpdf(c, m)
function logpdf(p::MixtureMassPrior, m::Real)
    acc = -Inf
    for (c, w) in zip(p.components, p.weights)
        acc = logaddexp(acc, log(w) + _component_logpdf(c, m))
    end
    return acc
end
function logpdf(p::MixtureMassPrior, m::Real, z::Real)
    acc = -Inf
    for (c, w) in zip(p.components, p.weights)
        acc = logaddexp(acc, log(w) + _component_logpdf(c, m, z))
    end
    return acc
end

"""
    DefaultSpinPrior(alpha_chi, beta_chi, sigma_t, csi_spin)

Spin prior on `(chi1, chi2, cos_t1, cos_t2)` matching the Python default:
Beta spin magnitudes and a mixture of isotropic and aligned tilt distributions.
"""
struct DefaultSpinPrior
    beta::BetaDistribution
    aligned::TruncatedGaussian
    csi_spin::Float64
end
DefaultSpinPrior(alpha_chi, beta_chi, sigma_t, csi_spin) =
    DefaultSpinPrior(BetaDistribution(alpha_chi, beta_chi), TruncatedGaussian(1, sigma_t, -1, 1), float(csi_spin))
function logpdf(p::DefaultSpinPrior, chi1, chi2, cos1, cos2)
    angular = logaddexp(log1p(-p.csi_spin) + log(0.25), log(p.csi_spin) + logpdf(p.aligned, cos1) + logpdf(p.aligned, cos2))
    return logpdf(p.beta, chi1) + logpdf(p.beta, chi2) + angular
end

"""
    GaussianSpinPrior(mu_chi_eff, sigma_chi_eff, mu_chi_p, sigma_chi_p, rho)

Bivariate Gaussian spin prior on `(chi_eff, chi_p)` with support
`chi_eff in [-1,1]`, `chi_p in [0,1]`.
"""
struct GaussianSpinPrior
    mu_eff::Float64
    sigma_eff::Float64
    mu_p::Float64
    sigma_p::Float64
    rho::Float64
end
function logpdf(p::GaussianSpinPrior, chi_eff, chi_p)
    -1 <= chi_eff <= 1 && 0 <= chi_p <= 1 || return -Inf
    z1 = (chi_eff - p.mu_eff) / p.sigma_eff
    z2 = (chi_p - p.mu_p) / p.sigma_p
    rho2 = 1 - p.rho^2
    return -log(2pi * p.sigma_eff * p.sigma_p * sqrt(rho2)) -
        (z1^2 - 2p.rho * z1 * z2 + z2^2) / (2rho2)
end

betadistro_muvar2ab(mu, var) = (mu * (mu * (1 - mu) / var - 1), (1 - mu) * (mu * (1 - mu) / var - 1))
betadistro_ab2muvar(a, b) = (a / (a + b), a * b / ((a + b)^2 * (a + b + 1)))

"""
    ParameterSpec(name, lower, upper; prior=:uniform, default=NaN, fixed=false, unit="", description="")

One entry in a sampler-facing parameter schema. Supported prior transforms are
`:uniform`, `:loguniform`, and `:fixed`.
"""
Base.@kwdef struct ParameterSpec
    name::Symbol
    lower::Float64
    upper::Float64
    prior::Symbol = :uniform
    default::Float64 = NaN
    fixed::Bool = false
    unit::String = ""
    description::String = ""
end
ParameterSpec(name::Symbol; lower, upper, prior::Symbol=:uniform, default=NaN, fixed::Bool=false, unit::AbstractString="", description::AbstractString="") =
    ParameterSpec(name, Float64(lower), Float64(upper), prior, Float64(default), fixed, String(unit), String(description))

"""
    ParameterSchema(specs)

Ordered schema for vector parameter interfaces. `theta[i]` corresponds to
`specs[i]`.
"""
struct ParameterSchema
    specs::Vector{ParameterSpec}
    index::Dict{Symbol,Int}
    function ParameterSchema(specs::Vector{ParameterSpec})
        names = [s.name for s in specs]
        length(unique(names)) == length(names) || throw(ArgumentError("parameter names must be unique"))
        return new(specs, Dict(name => i for (i, name) in pairs(names)))
    end
end
ParameterSchema(specs::ParameterSpec...) = ParameterSchema(collect(specs))
parameter_schema(x) = x.schema
Base.length(s::ParameterSchema) = length(s.specs)

"""
    prior_transform(schema, u)

Map a unit-cube vector to physical parameters using the schema. Matrix input is
interpreted as `nparameters x npoints`.
"""
function prior_transform(schema::ParameterSchema, u::AbstractVector{<:Real})
    length(u) == length(schema) || throw(ArgumentError("unit vector length does not match schema"))
    theta = Vector{Float64}(undef, length(schema))
    @inbounds for i in eachindex(schema.specs)
        spec = schema.specs[i]
        theta[i] = spec.fixed || spec.prior == :fixed ? spec.default :
            spec.prior == :uniform ? spec.lower + u[i] * (spec.upper - spec.lower) :
            spec.prior == :loguniform ? exp(log(spec.lower) + u[i] * (log(spec.upper) - log(spec.lower))) :
            throw(ArgumentError("unsupported prior transform $(spec.prior)"))
    end
    return theta
end
function prior_transform(schema::ParameterSchema, u::AbstractMatrix{<:Real})
    size(u, 1) == length(schema) || throw(ArgumentError("unit matrix must be nparameters x npoints"))
    out = Matrix{Float64}(undef, size(u))
    for j in axes(u, 2)
        out[:, j] = prior_transform(schema, view(u, :, j))
    end
    return out
end

"""
    unpack(schema, theta)

Return a `NamedTuple` of parameters from a schema-ordered vector.
"""
function unpack(schema::ParameterSchema, theta::AbstractVector{<:Real})
    length(theta) == length(schema) || throw(ArgumentError("parameter vector length does not match schema"))
    return NamedTuple{Tuple(s.name for s in schema.specs)}(Tuple(float.(theta)))
end

"""
    pack(schema, named)

Return a schema-ordered vector from a `NamedTuple` or dictionary-like object.
"""
function pack(schema::ParameterSchema, named)
    theta = Vector{Float64}(undef, length(schema))
    for (i, spec) in pairs(schema.specs)
        theta[i] = hasproperty(named, spec.name) ? Float64(getproperty(named, spec.name)) : Float64(named[spec.name])
    end
    return theta
end

end

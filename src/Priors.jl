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
    RedshiftConditionalMassDistribution,
    PairedMassDistribution,
    GeneralPairedMassDistribution,
    PiecewiseConstant2D,
    Bivariate2DGaussian,
    LowpassSmoothedProb,
    LowpassSmoothedProbEvolving,
    SmoothedPlusDipProb,
    AbsLuminosityPowerLawInMagnitude,
    absL_PL_inM,
    lowpass_filter,
    highpass_filter,
    notch_filter,
    mixed_linear_function,
    mixed_double_sigmoid_function,
    paired_massratio_dip,
    paired_massratio_dip_general,
    paired_massratio_bpl_dip_farah_2022,
    paired_bpl_triplepeak_dip,
    paired_massratio_bplmulti_dip,
    paired_massratio_bplmulti_dip_conditioned,
    bin_model_2d,
    PowerLawStationary,
    PowerLawLinear,
    GaussianStationary,
    GaussianLinear,
    MixtureMassPrior,
    RedshiftMixtureMassPrior,
    DefaultSpinPrior,
    GaussianComponentSpinPrior,
    EvolvingGaussianSpinPrior,
    BetaWindowGaussianSpinPrior,
    BetaWindowBetaSpinPrior,
    PSEOBGaussianPrior,
    ECOTotallyReflectiveSpinPrior,
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
logpdf(p::AbstractPrior, x::AbstractArray, z::AbstractArray) = map((a, b) -> logpdf(p, a, b), x, z)
logpdf(p::AbstractPrior, x::AbstractArray, y::AbstractArray, z::AbstractArray) =
    map((a, b, c) -> logpdf(p, a, b, c), x, y, z)
pdf(p::AbstractPrior, x, z) = exp.(logpdf(p, x, z))

_powerlaw_norm(min, max, alpha) =
    alpha == -1 ? log(max / min) : (max^(alpha + 1) - min^(alpha + 1)) / (alpha + 1)

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
        norm = _powerlaw_norm(float(min), float(max), a)
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
    return logaddexp(logpdf(p.pl1, x), logpdf(p.pl2, x) + log(p.ratio)) - log(p.norm)
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
    min::Float64
    max::Float64
end
function BrokenPowerLawMultiPeak(minpl, maxpl, alpha1, alpha2, b, lambda_g, lambda_low, mean_low, sigma_low, min_low, max_low, mean_high, sigma_high, min_high, max_high)
    return BrokenPowerLawMultiPeak(
        BrokenPowerLaw(minpl, maxpl, alpha1, alpha2, b),
        TruncatedGaussian(mean_low, sigma_low, min_low, max_low),
        TruncatedGaussian(mean_high, sigma_high, min_high, max_high),
        float(lambda_g),
        float(lambda_low),
        minimum((float(minpl), float(min_low), float(min_high))),
        maximum((float(maxpl), float(max_low), float(max_high))),
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
    min::Float64
    max::Float64
end
function BrokenPowerLawTripleMultiPeak(minpl, maxpl, alpha1, alpha2, b, lambda_g, lambda1, lambda2,
    mean1, sigma1, min1, max1, mean2, sigma2, min2, max2, mean3, sigma3, min3, max3)
    return BrokenPowerLawTripleMultiPeak(BrokenPowerLaw(minpl, maxpl, alpha1, alpha2, b),
        TruncatedGaussian(mean1, sigma1, min1, max1),
        TruncatedGaussian(mean2, sigma2, min2, max2),
        TruncatedGaussian(mean3, sigma3, min3, max3),
        float(lambda_g), float(lambda1), float(lambda2),
        minimum((float(minpl), float(min1), float(min2), float(min3))),
        maximum((float(maxpl), float(max1), float(max2), float(max3))))
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
    RedshiftConditionalMassDistribution(primary, secondary)

Two-dimensional distribution with a redshift-dependent primary mass density:
`p(m1, m2 | z) = p1(m1 | z) p2(m2) / CDF2(m1)` and support `m2 <= m1`.
This covers Python `CBC_rate_m1_given_redshift_m2` without a mutable wrapper
hierarchy.
"""
struct RedshiftConditionalMassDistribution{P1<:AbstractPrior,P2<:AbstractPrior} <: AbstractPrior
    primary::P1
    secondary::P2
end
function logpdf(p::RedshiftConditionalMassDistribution, m1::Real, m2::Real, z::Real)
    m2 <= m1 || return -Inf
    denom = cdf(p.secondary, m1)
    denom > 0 || return -Inf
    return logpdf(p.primary, m1, z) + logpdf(p.secondary, m2) - log(denom)
end
pdf(p::RedshiftConditionalMassDistribution, m1::Real, m2::Real, z::Real) =
    exp(logpdf(p, m1, m2, z))

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
    GeneralPairedMassDistribution(base, pairing_function)

Two-dimensional paired mass distribution proportional to
`base(m1) * base(m2) * pairing_function(m1, m2)`. The pairing function defines
the triangular support by returning zero when a point is not allowed.
"""
struct GeneralPairedMassDistribution{P<:AbstractPrior,F} <: AbstractPrior
    base::P
    pairing_function::F
    norm::Float64
    min::Float64
    max::Float64
end
function GeneralPairedMassDistribution(base::AbstractPrior, pairing_function; rtol::Real=1e-5)
    lo = getfield(base, :min)
    hi = getfield(base, :max)
    inner(m1) = quadgk(m2 -> pdf(base, m2) * pairing_function(m1, m2), lo, hi; rtol=rtol)[1]
    norm = quadgk(m1 -> pdf(base, m1) * inner(m1), lo, hi; rtol=rtol)[1]
    norm > 0 && isfinite(norm) || throw(ArgumentError("invalid paired-mass normalization"))
    return GeneralPairedMassDistribution(base, pairing_function, norm, lo, hi)
end
function logpdf(p::GeneralPairedMassDistribution, m1::Real, m2::Real)
    base_log = logpdf(p.base, m1) + logpdf(p.base, m2)
    isfinite(base_log) || return -Inf
    pair = p.pairing_function(m1, m2)
    pair > 0 && isfinite(pair) || return -Inf
    return base_log + log(pair) - log(p.norm)
end
pdf(p::GeneralPairedMassDistribution, m1::Real, m2::Real) = exp(logpdf(p, m1, m2))

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

"""
    bin_model_2d(min, max, weights)

Convenience constructor for Python `massprior_BinModel2d`-style triangular
piecewise-constant mass priors.
"""
bin_model_2d(min, max, weights) = PiecewiseConstant2D(min, max, weights)

_sigmoid_window(x, edge, delta, highpass::Bool) = delta == 0 ? 1.0 :
    highpass ? (x <= edge ? 0.0 : x >= edge + delta ? 1.0 :
        1 / (1 + exp(delta / (x - edge) + delta / (x - edge - delta)))) :
    (x >= edge ? 0.0 : x <= edge - delta ? 1.0 :
        1 / (1 + exp(delta / (edge - x) + delta / (edge - x - delta))))
_notch(x, left, dleft, right, dright, depth) =
    1 - depth * _sigmoid_window(x, left, dleft, true) * _sigmoid_window(x, right, dright, false)

"""
    highpass_filter(x, edge, delta)

Smooth turn-on window used by Python `_highpass_filter`.
"""
highpass_filter(x::Real, edge::Real, delta::Real) = _sigmoid_window(float(x), float(edge), float(delta), true)
highpass_filter(x, edge::Real, delta::Real) = highpass_filter.(x, edge, delta)

"""
    lowpass_filter(x, edge, delta)

Smooth turn-off window used by Python `_lowpass_filter`.
"""
lowpass_filter(x::Real, edge::Real, delta::Real) = _sigmoid_window(float(x), float(edge), float(delta), false)
lowpass_filter(x, edge::Real, delta::Real) = lowpass_filter.(x, edge, delta)

"""
    notch_filter(x, left, dleft, right, dright, depth)

Window `1 - depth * highpass_filter(x, left, dleft) * lowpass_filter(x, right, dright)`.
"""
notch_filter(x::Real, left::Real, dleft::Real, right::Real, dright::Real, depth::Real) =
    _notch(float(x), float(left), float(dleft), float(right), float(dright), float(depth))
notch_filter(x, left::Real, dleft::Real, right::Real, dright::Real, depth::Real) =
    notch_filter.(x, left, dleft, right, dright, depth)

"""
    mixed_linear_function(x, x0, x1)

Linear interpolation `(x1 - x0) * x + x0`.
"""
mixed_linear_function(x, x0, x1) = (x1 - x0) .* x .+ x0

"""
    mixed_double_sigmoid_function(x, xt, delta_xt, mix_x0, mix_x1)

Sigmoid transition from `mix_x0` to `mix_x1`, matching Python
`_mixed_double_sigmoid_function`.
"""
mixed_double_sigmoid_function(x, xt, delta_xt, mix_x0, mix_x1) =
    mix_x1 .+ (mix_x0 - mix_x1) ./ (1 .+ exp.((x .- xt) ./ delta_xt))

_gaussian_trunc_norm(lo, hi, mean, sigma) =
    0.5 * erf((hi - mean) / (sigma * sqrt(2))) - 0.5 * erf((lo - mean) / (sigma * sqrt(2)))

"""
    Bivariate2DGaussian(...)

Truncated bivariate Gaussian expressed as a truncated marginal in `x1` times a
truncated conditional Gaussian in `x2 | x1`, matching Python
`Bivariate2DGaussian`.
"""
struct Bivariate2DGaussian <: AbstractPrior
    x1min::Float64
    x1max::Float64
    x1mean::Float64
    x2min::Float64
    x2max::Float64
    x2mean::Float64
    x1variance::Float64
    x12covariance::Float64
    x2variance::Float64
    norm_marginal_1::Float64
end
function Bivariate2DGaussian(; x1min, x1max, x1mean, x2min, x2max, x2mean,
    x1variance, x12covariance, x2variance)
    x1min < x1max || throw(ArgumentError("x1min must be smaller than x1max"))
    x2min < x2max || throw(ArgumentError("x2min must be smaller than x2max"))
    x1variance > 0 && x2variance > 0 || throw(ArgumentError("variances must be positive"))
    condvar = x2variance - x12covariance^2 / x1variance
    condvar > 0 || throw(ArgumentError("conditional variance must be positive"))
    norm1 = _gaussian_trunc_norm(x1min, x1max, x1mean, sqrt(x1variance))
    norm1 > 0 || throw(ArgumentError("invalid x1 truncation normalization"))
    return Bivariate2DGaussian(float(x1min), float(x1max), float(x1mean), float(x2min), float(x2max),
        float(x2mean), float(x1variance), float(x12covariance), float(x2variance), norm1)
end
function logpdf(p::Bivariate2DGaussian, x1::Real, x2::Real)
    p.x1min <= x1 <= p.x1max && p.x2min <= x2 <= p.x2max || return -Inf
    marginal = -0.5 * log(2pi * p.x1variance) - 0.5 * (x1 - p.x1mean)^2 / p.x1variance -
        log(p.norm_marginal_1)
    mean2 = p.x2mean + (p.x12covariance / p.x1variance) * (x1 - p.x1mean)
    var2 = p.x2variance - p.x12covariance^2 / p.x1variance
    norm2 = _gaussian_trunc_norm(p.x2min, p.x2max, mean2, sqrt(var2))
    norm2 > 0 || return -Inf
    conditional = -0.5 * log(2pi * var2) - 0.5 * (x2 - mean2)^2 / var2 - log(norm2)
    return marginal + conditional
end
pdf(p::Bivariate2DGaussian, x1::Real, x2::Real) = exp(logpdf(p, x1, x2))
logpdf(p::Bivariate2DGaussian, x1::AbstractArray, x2::AbstractArray) = map((a, b) -> logpdf(p, a, b), x1, x2)
pdf(p::Bivariate2DGaussian, x1::AbstractArray, x2::AbstractArray) = exp.(logpdf(p, x1, x2))

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
    LowpassSmoothedProbEvolving(origin, delta_m)

Python-compatible low-mass smoothing wrapper for priors that may depend on
redshift. Its normalization follows Python `LowpassSmoothedProbEvolving`,
including the fixed 1000-point trapezoidal rule used in that implementation.
"""
struct LowpassSmoothedProbEvolving{P<:AbstractPrior} <: AbstractPrior
    origin::P
    delta::Float64
    min::Float64
    max::Float64
end
function LowpassSmoothedProbEvolving(origin::AbstractPrior, delta::Real)
    return LowpassSmoothedProbEvolving(origin, float(delta), getfield(origin, :min), getfield(origin, :max))
end
function _redshift_support_bounds(origin, z)
    if hasfield(typeof(origin), :mmin_z0) && hasfield(typeof(origin), :mmin_z1) &&
       hasfield(typeof(origin), :mmax_z0) && hasfield(typeof(origin), :mmax_z1)
        return getfield(origin, :mmin_z0) + getfield(origin, :mmin_z1) * z,
            getfield(origin, :mmax_z0) + getfield(origin, :mmax_z1) * z
    end
    return getfield(origin, :min), getfield(origin, :max)
end
function _trapezoid_integral(values::AbstractVector{<:Real}, xs::AbstractVector{<:Real})
    total = 0.0
    for i in 1:(length(xs) - 1)
        total += 0.5 * (values[i] + values[i + 1]) * (xs[i + 1] - xs[i])
    end
    return total
end
function _lowpass_evolving_norm(p::LowpassSmoothedProbEvolving, z=nothing)
    p.delta <= 0 && return 1.0
    lo, _ = z === nothing ? (p.min, p.max) : _redshift_support_bounds(p.origin, z)
    isfinite(lo) || throw(ArgumentError("LowpassSmoothedProbEvolving needs finite lower support"))
    xs = range(lo, lo + p.delta; length=1000)
    base = z === nothing ? [pdf(p.origin, x) for x in xs] : [exp(_component_logpdf(p.origin, x, z)) for x in xs]
    xvec = collect(xs)
    smoothed = base .* highpass_filter(xvec, lo, p.delta)
    integral_before = _trapezoid_integral(base, xvec)
    integral_now = _trapezoid_integral(smoothed, xvec)
    return 1 - integral_before + integral_now
end
function logpdf(p::LowpassSmoothedProbEvolving, x::Real)
    _inrange(x, p.min, p.max) || return -Inf
    w = highpass_filter(x, p.min, p.delta)
    w > 0 || return -Inf
    return logpdf(p.origin, x) + log(w) - log(_lowpass_evolving_norm(p))
end
function logpdf(p::LowpassSmoothedProbEvolving, x::Real, z::Real)
    lo, hi = _redshift_support_bounds(p.origin, z)
    _inrange(x, lo, hi) || return -Inf
    w = highpass_filter(x, lo, p.delta)
    w > 0 || return -Inf
    return _component_logpdf(p.origin, x, z) + log(w) - log(_lowpass_evolving_norm(p, z))
end
cdf(p::LowpassSmoothedProbEvolving, x::Real) =
    x <= p.min ? 0.0 : x >= p.max ? 1.0 : quadgk(t -> pdf(p, t), p.min, x; rtol=1e-5)[1]

"""
    AbsLuminosityPowerLawInMagnitude(Mmin, Mmax, alpha)

Absolute-magnitude prior equivalent to Python `absL_PL_inM`. It represents a
power law in luminosity, `p(L) ∝ L^alpha`, on the luminosity range implied by
`[Mmin, Mmax]`, and includes the magnitude-space Jacobian.
"""
struct AbsLuminosityPowerLawInMagnitude <: AbstractPrior
    min::Float64
    max::Float64
    alpha::Float64
    Lmin::Float64
    Lmax::Float64
    luminosity_prior::PowerLaw
    luminosity_cdf_prior::PowerLaw
    extrafact::Float64
end
function AbsLuminosityPowerLawInMagnitude(Mmin::Real, Mmax::Real, alpha::Real)
    Mmin < Mmax || throw(ArgumentError("AbsLuminosityPowerLawInMagnitude requires Mmin < Mmax"))
    a = float(alpha)
    Lmax = 3.0128e28 * 10.0^(-0.4 * float(Mmin))
    Lmin = 3.0128e28 * 10.0^(-0.4 * float(Mmax))
    Lmin < Lmax || throw(ArgumentError("invalid luminosity range"))
    lum_prior = PowerLaw(Lmin, Lmax, a + 1)
    lum_cdf_prior = PowerLaw(Lmin, Lmax, a)
    extrafact = 0.4 * log(10) * _powerlaw_norm(Lmin, Lmax, a + 1) / _powerlaw_norm(Lmin, Lmax, a)
    return AbsLuminosityPowerLawInMagnitude(float(Mmin), float(Mmax), a, Lmin, Lmax, lum_prior, lum_cdf_prior, extrafact)
end
function logpdf(p::AbsLuminosityPowerLawInMagnitude, M::Real)
    _inrange(M, p.min, p.max) || return -Inf
    L = 3.0128e28 * 10.0^(-0.4 * M)
    return logpdf(p.luminosity_prior, L) + log(p.extrafact)
end
function cdf(p::AbsLuminosityPowerLawInMagnitude, M::Real)
    M <= p.min && return 0.0
    M >= p.max && return 1.0
    L = 3.0128e28 * 10.0^(-0.4 * M)
    return 1 - cdf(p.luminosity_cdf_prior, L)
end
const absL_PL_inM = AbsLuminosityPowerLawInMagnitude

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

function _dip_smoothed_base(base::AbstractPrior; bottomsmooth, topsmooth, leftdip, rightdip, leftdipsmooth, rightdipsmooth, deep)
    return SmoothedPlusDipProb(base, bottomsmooth, topsmooth, leftdip, rightdip, leftdipsmooth, rightdipsmooth, deep)
end

"""
    paired_massratio_dip(base; beta, bottomsmooth, topsmooth, leftdip, rightdip, leftdipsmooth, rightdipsmooth, deep)

Python `m1m2_paired_massratio_dip` equivalent: smooth/dip a one-dimensional
base mass prior and pair binaries with `q^beta`.
"""
function paired_massratio_dip(base::AbstractPrior; beta, bottomsmooth, topsmooth, leftdip, rightdip, leftdipsmooth, rightdipsmooth, deep)
    smoothed = _dip_smoothed_base(base; bottomsmooth, topsmooth, leftdip, rightdip, leftdipsmooth, rightdipsmooth, deep)
    return GeneralPairedMassDistribution(smoothed, _piecewise_q_pairing(beta, beta, Inf))
end

function _piecewise_q_pairing(beta_bottom, beta_top, break_mass)
    return function (m1, m2)
        q = m2 / m1
        0 < q <= 1 || return 0.0
        return m2 <= break_mass ? q^beta_bottom : q^beta_top
    end
end

"""
    paired_massratio_dip_general(base; beta_bottom, beta_top, ...)

Python `m1m2_paired_massratio_dip_general` equivalent. The pairing exponent
switches from `beta_bottom` to `beta_top` at `rightdip`, matching the Python
wrapper's default split.
"""
function paired_massratio_dip_general(base::AbstractPrior; beta_bottom, beta_top, bottomsmooth, topsmooth,
    leftdip, rightdip, leftdipsmooth, rightdipsmooth, deep, break_mass=rightdip)
    smoothed = _dip_smoothed_base(base; bottomsmooth, topsmooth, leftdip, rightdip, leftdipsmooth, rightdipsmooth, deep)
    return GeneralPairedMassDistribution(smoothed, _piecewise_q_pairing(beta_bottom, beta_top, break_mass))
end

"""
    paired_massratio_bpl_dip_farah_2022(; alpha_1, alpha_2, mmin, mmax, ...)

Python `m1m2_paired_massratio_bpl_dip_farah_2022` equivalent. The power-law
slopes follow the Python wrapper convention, so the underlying
`BrokenPowerLaw` receives `-alpha_1` and `-alpha_2`.
"""
function paired_massratio_bpl_dip_farah_2022(; alpha_1, alpha_2, mmin, mmax, beta_bottom, beta_top,
    bottomsmooth, topsmooth, leftdip, rightdip, leftdipsmooth, rightdipsmooth, deep)
    b = (leftdip - mmin) / (mmax - mmin)
    0 < b < 1 || throw(ArgumentError("leftdip must define a break inside [mmin, mmax]"))
    base = BrokenPowerLaw(mmin, mmax, -alpha_1, -alpha_2, b)
    smoothed = _dip_smoothed_base(base; bottomsmooth, topsmooth, leftdip, rightdip, leftdipsmooth, rightdipsmooth, deep)
    return GeneralPairedMassDistribution(smoothed, _piecewise_q_pairing(beta_bottom, beta_top, 5.0))
end

function _dip_break_fraction(mmin, mmax, leftdip, rightdip, leftdipsmooth, rightdipsmooth)
    mbreak_ns = leftdip + leftdipsmooth
    mbreak_bh = rightdip - rightdipsmooth
    mbreak = 0.5 * (mbreak_ns + mbreak_bh)
    b = (mbreak - mmin) / (mmax - mmin)
    0 < b < 1 || throw(ArgumentError("dip-derived break must lie inside [mmin, mmax]"))
    return b, mbreak
end

"""
    paired_massratio_bplmulti_dip(; alpha_1, alpha_2, mmin, mmax, ...)

Python `m1m2_paired_massratio_bplmulti_dip` equivalent: a broken-power-law
two-peak mass prior with a dip smoother and a secondary-mass split in the
mass-ratio pairing exponent.
"""
function paired_massratio_bplmulti_dip(; alpha_1, alpha_2, mmin, mmax, beta_bottom, beta_top,
    bottomsmooth, topsmooth, leftdip, rightdip, leftdipsmooth, rightdipsmooth, deep,
    mu_g_low, sigma_g_low, lambda_g_low, mu_g_high, sigma_g_high, lambda_g)
    b, mbreak = _dip_break_fraction(mmin, mmax, leftdip, rightdip, leftdipsmooth, rightdipsmooth)
    base = BrokenPowerLawMultiPeak(mmin, mmax, -alpha_1, -alpha_2, b, lambda_g, lambda_g_low,
        mu_g_low, sigma_g_low, mmin, mu_g_low + 5sigma_g_low,
        mu_g_high, sigma_g_high, mmin, mu_g_high + 5sigma_g_high)
    smoothed = _dip_smoothed_base(base; bottomsmooth, topsmooth, leftdip, rightdip, leftdipsmooth, rightdipsmooth, deep)
    return GeneralPairedMassDistribution(smoothed, _piecewise_q_pairing(beta_bottom, beta_top, mbreak))
end

"""
    paired_bpl_triplepeak_dip(; alpha_1, alpha_2, mmin, mmax, ...)

Python `m1m2_paired_bpl_triplepeak_dip` equivalent using the native
`BrokenPowerLawTripleMultiPeak` mass prior.
"""
function paired_bpl_triplepeak_dip(; alpha_1, alpha_2, mmin, mmax, beta_bottom, beta_top,
    bottomsmooth, topsmooth, leftdip, rightdip, leftdipsmooth, rightdipsmooth, deep,
    mu_g_1, sigma_g_1, lambda_g, mu_g_2, sigma_g_2, lambda_1, mu_g_3, sigma_g_3, lambda_2)
    b, mbreak = _dip_break_fraction(mmin, mmax, leftdip, rightdip, leftdipsmooth, rightdipsmooth)
    base = BrokenPowerLawTripleMultiPeak(mmin, mmax, -alpha_1, -alpha_2, b, lambda_g, lambda_1, lambda_2,
        mu_g_1, sigma_g_1, mmin, mu_g_1 + 5sigma_g_1,
        mu_g_2, sigma_g_2, mmin, mu_g_2 + 5sigma_g_2,
        mu_g_3, sigma_g_3, mmin, mu_g_3 + 5sigma_g_3)
    smoothed = _dip_smoothed_base(base; bottomsmooth, topsmooth, leftdip, rightdip, leftdipsmooth, rightdipsmooth, deep)
    return GeneralPairedMassDistribution(smoothed, _piecewise_q_pairing(beta_bottom, beta_top, mbreak))
end

"""
    paired_massratio_bplmulti_dip_conditioned(; alpha_1, alpha_2, mmin, mmax, ...)

Python `m1m2_paired_massratio_bplmulti_dip_conditioned` equivalent. The
primary mass follows the smoothed/dipped multi-peak model while the secondary
mass is conditionally distributed as a low-pass-smoothed broken power law.
"""
function paired_massratio_bplmulti_dip_conditioned(; alpha_1, alpha_2, mmin, mmax, beta_bottom, beta_top,
    bottomsmooth, topsmooth, leftdip, rightdip, leftdipsmooth, rightdipsmooth, deep,
    mu_g_low, sigma_g_low, lambda_g_low, mu_g_high, sigma_g_high, lambda_g)
    b, _ = _dip_break_fraction(mmin, mmax, leftdip, rightdip, leftdipsmooth, rightdipsmooth)
    primary_base = BrokenPowerLawMultiPeak(mmin, mmax, -alpha_1, -alpha_2, b, lambda_g, lambda_g_low,
        mu_g_low, sigma_g_low, mmin, mu_g_low + 5sigma_g_low,
        mu_g_high, sigma_g_high, mmin, mu_g_high + 5sigma_g_high)
    primary = _dip_smoothed_base(primary_base; bottomsmooth, topsmooth, leftdip, rightdip, leftdipsmooth, rightdipsmooth, deep)
    secondary = LowpassSmoothedProb(BrokenPowerLaw(mmin, mmax, beta_bottom, beta_top, b), bottomsmooth)
    return ConditionalMassDistribution(primary, secondary)
end

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
    RedshiftMixtureMassPrior(components, weight_function)

Redshift-dependent mixture of stationary or redshift-dependent mass priors.
`weight_function(z)` must return one non-negative weight per component, with
weights summing to one. This represents Python's redshift-linear mixture
wrapper families while keeping the Julia API composable.
"""
struct RedshiftMixtureMassPrior{C<:Tuple,F} <: AbstractPrior
    components::C
    weight_function::F
    min::Float64
    max::Float64
end
function RedshiftMixtureMassPrior(components::Tuple, weight_function)
    isempty(components) && throw(ArgumentError("components must not be empty"))
    mins = [hasfield(typeof(c), :min) ? getfield(c, :min) : -Inf for c in components]
    maxs = [hasfield(typeof(c), :max) ? getfield(c, :max) : Inf for c in components]
    return RedshiftMixtureMassPrior(components, weight_function, minimum(mins), maximum(maxs))
end
RedshiftMixtureMassPrior(components::AbstractVector, weight_function) =
    RedshiftMixtureMassPrior(Tuple(components), weight_function)
function _validated_redshift_weights(p::RedshiftMixtureMassPrior, z)
    weights = collect(Float64, p.weight_function(z))
    length(weights) == length(p.components) || throw(ArgumentError("weight_function returned the wrong number of weights"))
    all(isfinite, weights) && all(>=(0), weights) || return nothing
    abs(sum(weights) - 1) <= 1e-10 || return nothing
    return weights
end
function logpdf(p::RedshiftMixtureMassPrior, m::Real, z::Real)
    weights = _validated_redshift_weights(p, z)
    weights === nothing && return -Inf
    acc = -Inf
    for (c, w) in zip(p.components, weights)
        w > 0 || continue
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
_component_spin_angular_logpdf(aligned::TruncatedGaussian, csi_spin, cos1, cos2) =
    logaddexp(log1p(-csi_spin) + log(0.25), log(csi_spin) + logpdf(aligned, cos1) + logpdf(aligned, cos2))
function logpdf(p::DefaultSpinPrior, chi1, chi2, cos1, cos2)
    angular = _component_spin_angular_logpdf(p.aligned, p.csi_spin, cos1, cos2)
    return logpdf(p.beta, chi1) + logpdf(p.beta, chi2) + angular
end

"""
    GaussianComponentSpinPrior(mu_chi_1, mu_chi_2, sigma_chi_1, sigma_chi_2, sigma_t, csi_spin)

Component-spin analogue of Python `spinprior_default_gaussian`.
"""
struct GaussianComponentSpinPrior
    g1::TruncatedGaussian
    g2::TruncatedGaussian
    aligned::TruncatedGaussian
    csi_spin::Float64
end
GaussianComponentSpinPrior(mu_chi_1, mu_chi_2, sigma_chi_1, sigma_chi_2, sigma_t, csi_spin) =
    GaussianComponentSpinPrior(TruncatedGaussian(mu_chi_1, sigma_chi_1, 0, 1),
        TruncatedGaussian(mu_chi_2, sigma_chi_2, 0, 1), TruncatedGaussian(1, sigma_t, -1, 1), float(csi_spin))
function logpdf(p::GaussianComponentSpinPrior, chi1, chi2, cos1, cos2)
    return logpdf(p.g1, chi1) + logpdf(p.g2, chi2) + _component_spin_angular_logpdf(p.aligned, p.csi_spin, cos1, cos2)
end

"""
    EvolvingGaussianSpinPrior(mu_chi, sigma_chi, mu_dot, sigma_dot, sigma_t, csi_spin)

Mass-dependent Gaussian component-spin prior matching Python
`spinprior_default_evolving_gaussian`.
"""
struct EvolvingGaussianSpinPrior
    mu_chi::Float64
    sigma_chi::Float64
    mu_dot::Float64
    sigma_dot::Float64
    sigma_t::Float64
    aligned::TruncatedGaussian
    csi_spin::Float64
end
EvolvingGaussianSpinPrior(mu_chi, sigma_chi, mu_dot, sigma_dot, sigma_t, csi_spin) =
    EvolvingGaussianSpinPrior(float(mu_chi), float(sigma_chi), float(mu_dot), float(sigma_dot),
        float(sigma_t), TruncatedGaussian(1, sigma_t, -1, 1), float(csi_spin))
function logpdf(p::EvolvingGaussianSpinPrior, chi1, chi2, cos1, cos2, mass1_source, mass2_source)
    sigma1 = p.sigma_chi + p.sigma_dot * mass1_source
    sigma2 = p.sigma_chi + p.sigma_dot * mass2_source
    sigma1 > 0 && sigma2 > 0 || return -Inf
    g1 = TruncatedGaussian(p.mu_chi + p.mu_dot * mass1_source, sigma1, 0, 1)
    g2 = TruncatedGaussian(p.mu_chi + p.mu_dot * mass2_source, sigma2, 0, 1)
    return logpdf(g1, chi1) + logpdf(g2, chi2) + _component_spin_angular_logpdf(p.aligned, p.csi_spin, cos1, cos2)
end

"""
    BetaWindowGaussianSpinPrior(mt, delta_mt, mix_f, alpha_chi, beta_chi, mu_chi, sigma_chi, sigma_t, csi_spin)

Mass-window mixture between a beta spin-magnitude prior and a truncated
Gaussian spin-magnitude prior.
"""
struct BetaWindowGaussianSpinPrior
    mt::Float64
    delta_mt::Float64
    mix_f::Float64
    beta::BetaDistribution
    gaussian::TruncatedGaussian
    aligned::TruncatedGaussian
    csi_spin::Float64
end
function BetaWindowGaussianSpinPrior(mt, delta_mt, mix_f, alpha_chi, beta_chi, mu_chi, sigma_chi, sigma_t, csi_spin)
    alpha_chi > 1 && beta_chi > 1 || throw(ArgumentError("alpha_chi and beta_chi must be greater than 1"))
    return BetaWindowGaussianSpinPrior(float(mt), float(delta_mt), float(mix_f), BetaDistribution(alpha_chi, beta_chi),
        TruncatedGaussian(mu_chi, sigma_chi, 0, 1), TruncatedGaussian(1, sigma_t, -1, 1), float(csi_spin))
end
_spin_window_weight(p, mass) = mixed_double_sigmoid_function(mass, p.mix_f, 0.0, p.mt, p.delta_mt)
function _mixture_logpdf(weight, p1, p2, x)
    return logaddexp(log(weight) + logpdf(p1, x), log1p(-weight) + logpdf(p2, x))
end
function logpdf(p::BetaWindowGaussianSpinPrior, chi1, chi2, cos1, cos2, mass1_source, mass2_source)
    w1 = _spin_window_weight(p, mass1_source)
    w2 = _spin_window_weight(p, mass2_source)
    0 <= w1 <= 1 && 0 <= w2 <= 1 || return -Inf
    return _mixture_logpdf(w1, p.beta, p.gaussian, chi1) +
        _mixture_logpdf(w2, p.beta, p.gaussian, chi2) +
        _component_spin_angular_logpdf(p.aligned, p.csi_spin, cos1, cos2)
end

"""
    BetaWindowBetaSpinPrior(...)

Mass-window mixture between two beta spin-magnitude priors.
"""
struct BetaWindowBetaSpinPrior
    mt::Float64
    delta_mt::Float64
    mix_f::Float64
    beta_low::BetaDistribution
    beta_high::BetaDistribution
    aligned::TruncatedGaussian
    csi_spin::Float64
end
function BetaWindowBetaSpinPrior(mt, delta_mt, mix_f, alpha_chi_low, beta_chi_low,
    alpha_chi_high, beta_chi_high, sigma_t, csi_spin)
    alpha_chi_low > 1 && beta_chi_low > 1 && alpha_chi_high > 1 && beta_chi_high > 1 ||
        throw(ArgumentError("beta shape parameters must be greater than 1"))
    return BetaWindowBetaSpinPrior(float(mt), float(delta_mt), float(mix_f),
        BetaDistribution(alpha_chi_low, beta_chi_low), BetaDistribution(alpha_chi_high, beta_chi_high),
        TruncatedGaussian(1, sigma_t, -1, 1), float(csi_spin))
end
function logpdf(p::BetaWindowBetaSpinPrior, chi1, chi2, cos1, cos2, mass1_source, mass2_source)
    w1 = _spin_window_weight(p, mass1_source)
    w2 = _spin_window_weight(p, mass2_source)
    0 <= w1 <= 1 && 0 <= w2 <= 1 || return -Inf
    return _mixture_logpdf(w1, p.beta_low, p.beta_high, chi1) +
        _mixture_logpdf(w2, p.beta_low, p.beta_high, chi2) +
        _component_spin_angular_logpdf(p.aligned, p.csi_spin, cos1, cos2)
end

"""
    PSEOBGaussianPrior(mu_domega220, sigma_domega220, mu_dtau220, sigma_dtau220, rho_pseob)

Bivariate Gaussian prior for pSEOB ringdown deviations.
"""
struct PSEOBGaussianPrior
    pdf_evaluator::Bivariate2DGaussian
end
PSEOBGaussianPrior(mu_domega220, sigma_domega220, mu_dtau220, sigma_dtau220, rho_pseob) =
    PSEOBGaussianPrior(Bivariate2DGaussian(x1min=-10, x1max=10, x1mean=mu_domega220,
        x2min=-10, x2max=10, x2mean=mu_dtau220, x1variance=sigma_domega220^2,
        x12covariance=rho_pseob * sigma_domega220 * sigma_dtau220, x2variance=sigma_dtau220^2))
logpdf(p::PSEOBGaussianPrior, domega220, dtau220) = logpdf(p.pdf_evaluator, domega220, dtau220)

"""
    ECOTotallyReflectiveSpinPrior(alpha_chi, beta_chi, eps, f_eco, sigma_chi_eco; q=1)

Spin-magnitude prior for the Python `spinprior_ECOs_totally_reflective` model.
"""
struct ECOTotallyReflectiveSpinPrior
    q::Float64
    beta::BetaDistribution
    truncated_beta::TruncatedBetaDistribution
    truncated_gaussian::TruncatedGaussian
    chi_crit::Float64
    f_eco::Float64
    lambda_eco::Float64
end
_eco_chi_crit(eps, q) = pi * (1 + q) / (2abs(log(eps)))
function ECOTotallyReflectiveSpinPrior(alpha_chi, beta_chi, eps, f_eco, sigma_chi_eco; q=1.0)
    alpha_chi > 1 && beta_chi > 1 || throw(ArgumentError("alpha_chi and beta_chi must be greater than 1"))
    0 <= f_eco <= 1 || throw(ArgumentError("f_eco must lie in [0, 1]"))
    chi_crit = _eco_chi_crit(eps, q)
    beta = BetaDistribution(alpha_chi, beta_chi)
    truncated_beta = TruncatedBetaDistribution(alpha_chi, beta_chi, chi_crit)
    truncated_gaussian = TruncatedGaussian(chi_crit, sigma_chi_eco, 0, chi_crit)
    lambda_eco = 1 - cdf(beta, chi_crit)
    return ECOTotallyReflectiveSpinPrior(float(q), beta, truncated_beta, truncated_gaussian, chi_crit, float(f_eco), lambda_eco)
end
function _eco_spin_pdf(p::ECOTotallyReflectiveSpinPrior, chi)
    eco = (1 - p.lambda_eco) * pdf(p.truncated_beta, chi) + p.lambda_eco * pdf(p.truncated_gaussian, chi)
    return p.f_eco * eco + (1 - p.f_eco) * pdf(p.beta, chi)
end
function logpdf(p::ECOTotallyReflectiveSpinPrior, chi1, chi2)
    p1 = _eco_spin_pdf(p, chi1)
    p2 = _eco_spin_pdf(p, chi2)
    p1 > 0 && p2 > 0 || return -Inf
    return log(p1) + log(p2)
end

logpdf(p::DefaultSpinPrior, chi1::AbstractArray, chi2::AbstractArray, cos1::AbstractArray, cos2::AbstractArray) =
    map((a, b, c, d) -> logpdf(p, a, b, c, d), chi1, chi2, cos1, cos2)
logpdf(p::GaussianComponentSpinPrior, chi1::AbstractArray, chi2::AbstractArray, cos1::AbstractArray, cos2::AbstractArray) =
    map((a, b, c, d) -> logpdf(p, a, b, c, d), chi1, chi2, cos1, cos2)
logpdf(p::EvolvingGaussianSpinPrior, chi1::AbstractArray, chi2::AbstractArray, cos1::AbstractArray, cos2::AbstractArray,
    mass1_source::AbstractArray, mass2_source::AbstractArray) =
    map((a, b, c, d, e, f) -> logpdf(p, a, b, c, d, e, f), chi1, chi2, cos1, cos2, mass1_source, mass2_source)
logpdf(p::BetaWindowGaussianSpinPrior, chi1::AbstractArray, chi2::AbstractArray, cos1::AbstractArray, cos2::AbstractArray,
    mass1_source::AbstractArray, mass2_source::AbstractArray) =
    map((a, b, c, d, e, f) -> logpdf(p, a, b, c, d, e, f), chi1, chi2, cos1, cos2, mass1_source, mass2_source)
logpdf(p::BetaWindowBetaSpinPrior, chi1::AbstractArray, chi2::AbstractArray, cos1::AbstractArray, cos2::AbstractArray,
    mass1_source::AbstractArray, mass2_source::AbstractArray) =
    map((a, b, c, d, e, f) -> logpdf(p, a, b, c, d, e, f), chi1, chi2, cos1, cos2, mass1_source, mass2_source)
logpdf(p::ECOTotallyReflectiveSpinPrior, chi1::AbstractArray, chi2::AbstractArray) =
    map((a, b) -> logpdf(p, a, b), chi1, chi2)

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

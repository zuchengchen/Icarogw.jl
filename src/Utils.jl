module Utils

using Tables

export logaddexp,
    logsumexp,
    finite_or_neginf,
    ensure_vector,
    assert_same_length,
    check_posterior_samples_and_prior,
    check_bounds_1d,
    check_bounds_2d,
    check_bounds_1D,
    check_bounds_2D

"""
    logaddexp(a, b)

Return `log(exp(a) + exp(b))` with stable behavior for very small or infinite
inputs.
"""
logaddexp(a::Real, b::Real) = a == -Inf ? float(b) : b == -Inf ? float(a) :
    max(float(a), float(b)) + log1p(exp(-abs(float(a) - float(b))))

"""
    logsumexp(values)

Stable logarithm of a sum of exponentials. Empty or all-`-Inf` inputs return
`-Inf`.
"""
function logsumexp(values)
    m = maximum(values)
    isfinite(m) || return m
    s = zero(float(m))
    @inbounds for v in values
        s += exp(v - m)
    end
    return m + log(s)
end

finite_or_neginf(x::Real) = isfinite(x) ? float(x) : -Inf

ensure_vector(x::AbstractVector{<:Real}) = Float64.(x)
ensure_vector(x::Real) = [Float64(x)]

function assert_same_length(name::AbstractString, arrays...)
    isempty(arrays) && return nothing
    n = length(first(arrays))
    for a in arrays
        length(a) == n || throw(ArgumentError("$name columns must have the same length"))
    end
    return n
end

_column_pairs(data::AbstractDict) = pairs(data)
_column_pairs(data) = pairs(Tables.columntable(data))

"""
    check_posterior_samples_and_prior(posterior_samples, prior)

Validate that every posterior-sample column has the same length as `prior` and
that the prior contains no zero values. This ports the non-Condor Python
utility while raising `ArgumentError` instead of printing before failure.
"""
function check_posterior_samples_and_prior(posterior_samples, prior)
    prior_vec = collect(prior)
    nprior = length(prior_vec)
    for (name, values) in _column_pairs(posterior_samples)
        length(values) == nprior ||
            throw(ArgumentError("posterior column :$name has length $(length(values)); expected $nprior"))
    end
    zero_idx = findall(==(0.0), prior_vec)
    isempty(zero_idx) || throw(ArgumentError("prior contains zero values at indices $(collect(zero_idx))"))
    return nothing
end

"""
    check_bounds_1d(x, minval, maxval)

Return `true` where `x` lies outside `[minval, maxval]`, matching the CPU
semantics of Python `cupy_pal.check_bounds_1D`.
"""
check_bounds_1d(x::Real, minval::Real, maxval::Real) = x < minval || x > maxval
check_bounds_1d(x, minval::Real, maxval::Real) = (x .< minval) .| (x .> maxval)

"""
    check_bounds_2d(x1, x2, y)

Return `true` where `x1 < x2` or `y` is `NaN`, matching the CPU semantics of
Python `cupy_pal.check_bounds_2D`.
"""
check_bounds_2d(x1::Real, x2::Real, y::Real) = x1 < x2 || isnan(y)
check_bounds_2d(x1, x2, y) = (x1 .< x2) .| isnan.(y)

const check_bounds_1D = check_bounds_1d
const check_bounds_2D = check_bounds_2d

end

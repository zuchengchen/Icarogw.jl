module Utils

export logaddexp, logsumexp, finite_or_neginf, ensure_vector, assert_same_length

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

end

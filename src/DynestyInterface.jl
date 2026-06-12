module DynestyInterface

using ..DataContainers
using ..Likelihood
using ..Priors
using ..Rates

export dynesty_prior_transform, dynesty_loglikelihood, dynesty_problem

"""
    dynesty_prior_transform(schema)

Return a closure `u -> theta` compatible with `Dynesty.NestedSampler`.
"""
dynesty_prior_transform(schema::ParameterSchema) = u -> prior_transform(schema, u)

"""
    dynesty_loglikelihood(model_type, data; options=LikelihoodOptions(), kwargs...)

Return a closure `theta -> loglikelihood(model_type, data, theta; ...)`
compatible with `Dynesty.NestedSampler`.
"""
function dynesty_loglikelihood(model_type::Type{SimplePowerLawPopulation}, data::PopulationData; options::LikelihoodOptions=LikelihoodOptions(), kwargs...)
    return theta -> loglikelihood(model_type, data, theta; options, kwargs...)
end

"""
    dynesty_problem(model_type, data; schema=parameter_schema(model_type), options=LikelihoodOptions(), kwargs...)

Return `(loglikelihood, prior_transform, ndim)` for use with local
`../Dynesty.jl`. The core package does not import Dynesty.
"""
function dynesty_problem(model_type::Type{SimplePowerLawPopulation}, data::PopulationData; schema=parameter_schema(model_type), options::LikelihoodOptions=LikelihoodOptions(), kwargs...)
    return (
        loglikelihood=dynesty_loglikelihood(model_type, data; options, kwargs...),
        prior_transform=dynesty_prior_transform(schema),
        ndim=length(schema),
        schema=schema,
    )
end

end

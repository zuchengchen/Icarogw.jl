module Plotting

using Plots
using ..Priors
using ..Rates
using ..Likelihood

export plot_mass_prior, plot_redshift_rate, plot_likelihood_diagnostics

"""
    plot_mass_prior(prior; range=nothing, n=300)

Plot a one-dimensional mass prior with `Plots.jl`.
"""
function plot_mass_prior(prior::AbstractPrior; range=nothing, n=300)
    lo = range === nothing ? getfield(prior, :min) : first(range)
    hi = range === nothing ? getfield(prior, :max) : last(range)
    x = collect(LinRange(lo, hi, n))
    return plot(x, pdf(prior, x), xlabel="mass [Msun]", ylabel="density", label="mass prior")
end

"""
    plot_redshift_rate(rate; zmax=2, n=300)

Plot redshift-rate evolution `ψ(z)`.
"""
function plot_redshift_rate(rate::AbstractRedshiftRate; zmax=2.0, n=300)
    z = collect(LinRange(0, zmax, n))
    y = exp.(Rates.log_rate(rate, z))
    return plot(z, y, xlabel="redshift", ylabel="relative rate", label="rate")
end

"""
    plot_likelihood_diagnostics(diagnostics)

Plot per-event effective sample sizes and the injection effective sample size.
"""
function plot_likelihood_diagnostics(d::LikelihoodDiagnostics)
    x = collect(1:length(d.per_event_neff))
    plt = bar(x, d.per_event_neff, xlabel="event", ylabel="effective samples", label="posterior")
    hline!(plt, [d.injection_neff], label="injections")
    return plt
end

end

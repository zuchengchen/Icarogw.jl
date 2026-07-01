module Plotting

using Plots
using ..Priors
using ..Rates
using ..Likelihood
using ..Catalog
using Statistics: median

export plot_mass_prior,
    plot_redshift_rate,
    plot_likelihood_diagnostics,
    plot_counts_map,
    plot_mthr_map,
    plot_differential_effective_galaxies

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

function _plot_catalog_map(values; label="", kwargs...)
    x = collect(eachindex(values))
    return bar(x, values, xlabel="sky pixel", ylabel=label, label=label; kwargs...)
end

"""
    plot_counts_map(catalog; kwargs...)

Plot the per-pixel galaxy counts for a `GalaxyCatalog`.
"""
plot_counts_map(catalog::GalaxyCatalog; kwargs...) =
    _plot_catalog_map(return_counts_map(catalog); label="galaxies", kwargs...)

"""
    plot_mthr_map(catalog; kwargs...)

Plot the apparent-magnitude threshold map for a `GalaxyCatalog`.
"""
function plot_mthr_map(catalog::GalaxyCatalog; kwargs...)
    catalog.mthr_map === nothing && throw(ArgumentError("GalaxyCatalog does not contain a magnitude-threshold map"))
    return _plot_catalog_map(catalog.mthr_map; label="magnitude threshold", kwargs...)
end

function _percentile_rows(values::AbstractMatrix, p::Real)
    out = Vector{Float64}(undef, size(values, 1))
    for i in axes(values, 1)
        row = sort!(collect(view(values, i, :)))
        pos = 1 + (length(row) - 1) * float(p) / 100
        lo = floor(Int, pos)
        hi = ceil(Int, pos)
        out[i] = lo == hi ? row[lo] : row[lo] + (pos - lo) * (row[hi] - row[lo])
    end
    return out
end

"""
    plot_differential_effective_galaxies(z, diagnostics)

Plot the output of `check_differential_effective_galaxies`.
"""
function plot_differential_effective_galaxies(z, diagnostics)
    zvals = collect(z)
    total = diagnostics.catalog .+ diagnostics.background
    p = plot(zvals, vec(median(diagnostics.catalog; dims=2)), label="catalog",
        xlabel="redshift", ylabel="effective galaxies", yscale=:log10)
    plot!(p, zvals, vec(median(diagnostics.background; dims=2)), label="background")
    plot!(p, zvals, vec(median(total; dims=2)), label="sum")
    plot!(p, zvals, diagnostics.theoretical, label="theoretical", linestyle=:dash)
    plot!(p, zvals, _percentile_rows(total, 5), fillrange=_percentile_rows(total, 95),
        fillalpha=0.15, linealpha=0, label="")
    q = plot(zvals, vec(median(1 .- diagnostics.incompleteness; dims=2)),
        xlabel="redshift", ylabel="incompleteness", label="median")
    return plot(p, q, layout=(2, 1), link=:x)
end

end

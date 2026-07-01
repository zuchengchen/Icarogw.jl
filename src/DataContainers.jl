module DataContainers

using CSV
using DataFrames
using HDF5
using Tables
using ..Utils: assert_same_length

export PosteriorSamples,
    PosteriorSampleSet,
    InjectionSet,
    PopulationData,
    column,
    validate,
    subset_posterior_samples,
    subset_injections,
    read_posterior_csv,
    read_injections_csv,
    write_hdf5,
    read_posterior_hdf5,
    read_injections_hdf5

"""
    PosteriorSamples(data, prior; event_name=:event)

Array-backed posterior samples for one event. Required data columns for the
vertical-slice likelihood are `:mass_1`, `:mass_2`, and `:luminosity_distance`.
The `prior` vector is the detector-frame PE prior density evaluated for each
sample.
"""
struct PosteriorSamples
    event_name::Symbol
    names::Vector{Symbol}
    values::Matrix{Float64}
    prior::Vector{Float64}
end

function PosteriorSamples(data, prior=nothing; event_name::Symbol=:event)
    table = Tables.columntable(data)
    names = collect(Symbol.(keys(table)))
    vals = [Float64.(collect(table[n])) for n in names]
    if prior === nothing
        idx = findfirst(==(:prior), names)
        idx === nothing && throw(ArgumentError("PosteriorSamples requires a prior vector or :prior column"))
        prior_vec = vals[idx]
        deleteat!(names, idx)
        deleteat!(vals, idx)
    else
        prior_vec = Float64.(collect(prior))
    end
    assert_same_length("posterior", vals..., prior_vec)
    mat = Matrix{Float64}(undef, length(prior_vec), length(vals))
    for (j, v) in pairs(vals)
        mat[:, j] = v
    end
    ps = PosteriorSamples(event_name, names, mat, prior_vec)
    validate(ps)
    return ps
end
PosteriorSamples(data::Dict; event_name::Symbol=:event) = PosteriorSamples((; (Symbol(k) => v for (k, v) in data)...); event_name)

"""
    PosteriorSampleSet(events)

Collection of posterior samples. The constructor preserves event order.
"""
struct PosteriorSampleSet
    events::Vector{PosteriorSamples}
end
PosteriorSampleSet(events::PosteriorSamples...) = PosteriorSampleSet(collect(events))
function PosteriorSampleSet(events::Dict)
    return PosteriorSampleSet([PosteriorSamples(v; event_name=Symbol(k)) for (k, v) in events])
end
Base.length(ps::PosteriorSampleSet) = length(ps.events)

"""
    InjectionSet(data, prior; ntotal, Tobs=1)

Detected injection set for selection-effect correction. `ntotal` is the total
number of generated injections before detection cuts; `Tobs` is observing time
in years.
"""
struct InjectionSet
    names::Vector{Symbol}
    values::Matrix{Float64}
    prior::Vector{Float64}
    ntotal::Int
    Tobs::Float64
end
function InjectionSet(data, prior=nothing; ntotal::Integer, Tobs::Real=1.0)
    table = Tables.columntable(data)
    names = collect(Symbol.(keys(table)))
    vals = [Float64.(collect(table[n])) for n in names]
    if prior === nothing
        idx = findfirst(==(:prior), names)
        idx === nothing && throw(ArgumentError("InjectionSet requires a prior vector or :prior column"))
        prior_vec = vals[idx]
        deleteat!(names, idx)
        deleteat!(vals, idx)
    else
        prior_vec = Float64.(collect(prior))
    end
    assert_same_length("injection", vals..., prior_vec)
    ntotal >= length(prior_vec) || throw(ArgumentError("ntotal must be >= detected injection count"))
    mat = Matrix{Float64}(undef, length(prior_vec), length(vals))
    for (j, v) in pairs(vals)
        mat[:, j] = v
    end
    inj = InjectionSet(names, mat, prior_vec, Int(ntotal), Float64(Tobs))
    validate(inj)
    return inj
end

"""
    PopulationData(posteriors, injections)

Container passed to population likelihood functions.
"""
struct PopulationData
    posteriors::PosteriorSampleSet
    injections::InjectionSet
end

function column_index(names::Vector{Symbol}, name::Symbol)
    idx = findfirst(==(name), names)
    idx === nothing && throw(ArgumentError("missing column :$name"))
    return idx
end

"""
    column(container, name)

Return a zero-copy vector view of a named container column.
"""
column(ps::PosteriorSamples, name::Symbol) = view(ps.values, :, column_index(ps.names, name))
column(inj::InjectionSet, name::Symbol) = view(inj.values, :, column_index(inj.names, name))

function _row_indices(n::Integer, selector)
    if selector isa AbstractVector{Bool}
        length(selector) == n || throw(ArgumentError("Boolean selector length $(length(selector)) does not match row count $n"))
        return findall(selector)
    end
    idx = Int.(collect(selector))
    all(i -> 1 <= i <= n, idx) || throw(ArgumentError("row indices must lie in 1:$n"))
    return idx
end

"""
    subset_posterior_samples(samples, selector)

Return a row-subset of one posterior-sample container. `selector` may be a
Boolean mask or integer indices.
"""
function subset_posterior_samples(ps::PosteriorSamples, selector)
    idx = _row_indices(length(ps.prior), selector)
    return PosteriorSamples(ps.event_name, copy(ps.names), Matrix(ps.values[idx, :]), ps.prior[idx]) |> validate
end

"""
    subset_injections(injections, selector)

Return a detected-injection subset while preserving `ntotal` and `Tobs`.
`selector` may be a Boolean mask or integer indices.
"""
function subset_injections(inj::InjectionSet, selector)
    idx = _row_indices(length(inj.prior), selector)
    return InjectionSet(copy(inj.names), Matrix(inj.values[idx, :]), inj.prior[idx], inj.ntotal, inj.Tobs) |> validate
end

"""
    validate(container)

Check required columns, finite values, positive priors, and length consistency.
"""
function validate(ps::PosteriorSamples)
    !isempty(ps.names) || throw(ArgumentError("posterior $(ps.event_name) must contain at least one data column"))
    size(ps.values, 1) == length(ps.prior) || throw(ArgumentError("posterior prior length mismatch"))
    all(isfinite, ps.values) || throw(ArgumentError("posterior contains non-finite values"))
    all(>(0), ps.prior) || throw(ArgumentError("posterior prior must be positive"))
    return ps
end
function validate(inj::InjectionSet)
    !isempty(inj.names) || throw(ArgumentError("injections must contain at least one data column"))
    size(inj.values, 1) == length(inj.prior) || throw(ArgumentError("injection prior length mismatch"))
    all(isfinite, inj.values) || throw(ArgumentError("injections contain non-finite values"))
    all(>(0), inj.prior) || throw(ArgumentError("injection prior must be positive"))
    return inj
end
validate(data::PopulationData) = (foreach(validate, data.posteriors.events); validate(data.injections); data)

"""
    read_posterior_csv(path; event_name=:event)

Read one event posterior from CSV. The file must contain `prior` or pass a prior
vector after reading manually.
"""
read_posterior_csv(path; event_name::Symbol=:event) = PosteriorSamples(CSV.File(path) |> DataFrame; event_name)

"""
    read_injections_csv(path; ntotal, Tobs=1)

Read detected injections from CSV.
"""
read_injections_csv(path; ntotal::Integer, Tobs::Real=1.0) =
    InjectionSet(CSV.File(path) |> DataFrame; ntotal, Tobs)

"""
    write_hdf5(path, object)

Write posterior or injection containers to HDF5 using simple datasets. This is
intended for Julia-native analysis files, not Python pickle compatibility.
"""
function write_hdf5(path::AbstractString, ps::PosteriorSamples)
    h5open(path, "w") do h
        write(h, "names", String.(ps.names))
        write(h, "values", ps.values)
        write(h, "prior", ps.prior)
        attrs(h)["event_name"] = String(ps.event_name)
    end
    return path
end
function write_hdf5(path::AbstractString, inj::InjectionSet)
    h5open(path, "w") do h
        write(h, "names", String.(inj.names))
        write(h, "values", inj.values)
        write(h, "prior", inj.prior)
        attrs(h)["ntotal"] = inj.ntotal
        attrs(h)["Tobs"] = inj.Tobs
    end
    return path
end

"""
    read_posterior_hdf5(path)

Read an HDF5 posterior written by `write_hdf5`.
"""
function read_posterior_hdf5(path::AbstractString)
    h5open(path, "r") do h
        names = Symbol.(read(h, "names"))
        values = Matrix{Float64}(read(h, "values"))
        prior = Vector{Float64}(read(h, "prior"))
        hattrs = attrs(h)
        event_name = Symbol(haskey(hattrs, "event_name") ? hattrs["event_name"] : "event")
        return PosteriorSamples(event_name, names, values, prior) |> validate
    end
end

"""
    read_injections_hdf5(path)

Read an HDF5 injection set written by `write_hdf5`.
"""
function read_injections_hdf5(path::AbstractString)
    h5open(path, "r") do h
        names = Symbol.(read(h, "names"))
        values = Matrix{Float64}(read(h, "values"))
        prior = Vector{Float64}(read(h, "prior"))
        hattrs = attrs(h)
        ntotal = Int(hattrs["ntotal"])
        Tobs = Float64(hattrs["Tobs"])
        return InjectionSet(names, values, prior, ntotal, Tobs) |> validate
    end
end

end

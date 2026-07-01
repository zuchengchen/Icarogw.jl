module Stochastic

using CSV
using DataFrames
using HDF5
using Random
using ..Priors
using ..Rates
using ..Cosmology
using ..DataContainers
using ..Likelihood

import ..Priors: logpdf, pdf
import ..Rates: log_rate

export PNVelocityPowers,
    OmegaGWWeights,
    StochasticData,
    StochasticDiagnostics,
    pn_velocity_powers,
    dedf,
    precompute_omega_weights,
    read_stochastic_csv,
    write_stochastic_hdf5,
    read_stochastic_hdf5,
    spectral_siren_omega_gw,
    stochastic_loglikelihood,
    joint_loglikelihood,
    stochastic_planned

const SI_M_SUN = 1.988409870698051e30
const SI_G = 6.6743e-11
const SI_C = 299792458.0
const SI_KPC = 3.0856775814913675e19
const YEAR_SECONDS = 365.0 * 24.0 * 3600.0
const MSUN_TO_SECONDS = SI_M_SUN * SI_G / SI_C^3

"""
    PNVelocityPowers(v1, v2, v3)

Post-Newtonian velocity powers used by the Ajith waveform fits. This mirrors
Python `v(Mtot, f)` but gives the intermediate values a descriptive type.
"""
struct PNVelocityPowers
    v1::Float64
    v2::Float64
    v3::Float64
end

"""
    OmegaGWWeights

Monte-Carlo workspace for stochastic-background spectral-siren calculations.
Fields are source-frame masses, redshifts, proposal densities, optional spin
proposal density, and an `N x nfreq` matrix of emitted spectra.
"""
struct OmegaGWWeights
    frequencies::Vector{Float64}
    m1s::Vector{Float64}
    m2s::Vector{Float64}
    redshifts::Vector{Float64}
    p_m1::Vector{Float64}
    p_m2::Vector{Float64}
    p_z::Vector{Float64}
    dEdfs::Matrix{Float64}
    p_chi12::Union{Nothing,Float64}
end

Base.length(w::OmegaGWWeights) = length(w.m1s)

"""
    StochasticData(frequencies, Cf, sigma2s; reference_H0=100)

Data container for stochastic-background likelihoods. `Cf` and `sigma2s` are
the cross-correlation estimate and variance normalized at `reference_H0`,
matching Python `Stochastic_likelihood_only`.
"""
struct StochasticData
    frequencies::Vector{Float64}
    Cf::Vector{Float64}
    sigma2s::Vector{Float64}
    reference_H0::Float64
end
function StochasticData(frequencies, Cf, sigma2s; reference_H0::Real=100.0)
    freqs = Float64.(collect(frequencies))
    cf = Float64.(collect(Cf))
    sig = Float64.(collect(sigma2s))
    length(freqs) == length(cf) == length(sig) ||
        throw(ArgumentError("frequencies, Cf, and sigma2s must have the same length"))
    all(isfinite, freqs) && all(isfinite, cf) && all(isfinite, sig) ||
        throw(ArgumentError("stochastic data must be finite"))
    all(>(0), sig) || throw(ArgumentError("sigma2s must be positive"))
    return StochasticData(freqs, cf, sig, float(reference_H0))
end

function _first_present(names, candidates)
    for candidate in candidates
        candidate in names && return candidate
    end
    return nothing
end

function _table_column(table::DataFrame, candidates, label)
    idx = _first_present(Symbol.(names(table)), candidates)
    idx === nothing && throw(ArgumentError("stochastic data is missing $label column; accepted names: $(join(String.(candidates), ", "))"))
    return table[!, idx]
end

"""
    read_stochastic_csv(path; reference_H0=100)

Read stochastic-background data from CSV. Accepted columns are Python-style
`freqs`, `Cf`, `sigma2s`, or Julia-style `frequency`, `cf`, `sigma2`.
"""
function read_stochastic_csv(path::AbstractString; reference_H0::Real=100.0)
    table = CSV.File(path) |> DataFrame
    freqs = _table_column(table, (:freqs, :freq, :frequency, :frequencies), "frequency")
    cf = _table_column(table, (:Cf, :cf), "Cf")
    sigma2s = _table_column(table, (:sigma2s, :sigma2, :variance), "sigma2s")
    return StochasticData(freqs, cf, sigma2s; reference_H0)
end

"""
    write_stochastic_hdf5(path, data)

Write `StochasticData` to HDF5 datasets `freqs`, `Cf`, and `sigma2s`, with
`reference_H0` stored as a file attribute.
"""
function write_stochastic_hdf5(path::AbstractString, data::StochasticData)
    h5open(path, "w") do h
        write(h, "freqs", data.frequencies)
        write(h, "Cf", data.Cf)
        write(h, "sigma2s", data.sigma2s)
        attrs(h)["reference_H0"] = data.reference_H0
    end
    return path
end

"""
    read_stochastic_hdf5(path; reference_H0=nothing)

Read stochastic-background data from HDF5 datasets `freqs`, `Cf`, and
`sigma2s`. When `reference_H0` is not provided, the file attribute is used if
present, otherwise Python's `100` convention is assumed.
"""
function read_stochastic_hdf5(path::AbstractString; reference_H0=nothing)
    h5open(path, "r") do h
        freqs = read(h, "freqs")
        cf = read(h, "Cf")
        sigma2s = read(h, "sigma2s")
        hattrs = attrs(h)
        ref = reference_H0 === nothing ?
            (haskey(hattrs, "reference_H0") ? Float64(hattrs["reference_H0"]) : 100.0) :
            Float64(reference_H0)
        return StochasticData(freqs, cf, sigma2s; reference_H0=ref)
    end
end

"""
    StochasticDiagnostics

Diagnostics returned by `spectral_siren_omega_gw(...; return_diagnostics=true)`.
"""
Base.@kwdef struct StochasticDiagnostics
    nweights::Int
    nfrequencies::Int
    max_weight::Float64
    min_weight::Float64
    has_nan::Bool
    has_inf::Bool
end

"""
    pn_velocity_powers(Mtot, f)

Return `(pi * G*M_sun/c^3 * f * Mtot)^(1/3, 2/3, 1)` for total source-frame
mass `Mtot` in solar masses and frequency `f` in Hz.
"""
function pn_velocity_powers(Mtot::Real, f::Real)
    base = pi * MSUN_TO_SECONDS * Float64(f) * Float64(Mtot)
    return PNVelocityPowers(base^(1 / 3), base^(2 / 3), base)
end

function _dot3(a::NTuple{3,Float64}, v::PNVelocityPowers)
    return a[1] * v.v1 + a[2] * v.v2 + a[3] * v.v3
end

"""
    dedf(Mtot, freqs; eta=0.25, inspiral_only=false, pn=true, chi=0)

Energy spectrum radiated by a compact binary, matching Python `dEdf` in
`omega_gw.py`/`stochastic.py`. `Mtot` is in solar masses and frequencies are in
Hz. The return value is in SI units used by the Python reference.
"""
function dedf(Mtot::Real, freqs; eta::Real=0.25, inspiral_only::Bool=false, pn::Bool=true, chi::Real=0.0)
    M = Float64(Mtot)
    eta = Float64(eta)
    chi = Float64(chi)
    out = zeros(Float64, length(freqs))
    if inspiral_only
        fmerge = 2 * SI_C^3 / (6 * sqrt(6) * 2pi * SI_G * M * SI_M_SUN)
        for (i, fraw) in pairs(freqs)
            f = Float64(fraw)
            out[i] = f < fmerge ? f^(-1 / 3) : 0.0
        end
    elseif pn
        eta_arr = (eta, eta^2, eta^3)
        chi_arr = (1.0, chi, chi^2)
        fM = ((0.6437, 0.827, -0.2706), (-0.05822, -3.935, 0.0), (-7.092, 0.0, 0.0))
        fR = ((0.1469, -0.1228, -0.02609), (-0.0249, 0.1701, 0.0), (2.325, 0.0, 0.0))
        fC = ((-0.1331, -0.08172, 0.1451), (-0.2714, 0.1279, 0.0), (4.922, 0.0, 0.0))
        sig = ((-0.4098, -0.03523, 0.1008), (1.829, -0.02017, 0.0), (-2.87, 0.0, 0.0))
        correction(mat) = sum(eta_arr[i] * mat[i][j] * chi_arr[j] for i in 1:3, j in 1:3)
        fmerge = (1 - 4.455 * (1 - chi)^0.217 + 3.521 * (1 - chi)^0.26 + correction(fM)) / (pi * M * MSUN_TO_SECONDS)
        fring = (0.5 - 0.315 * (1 - chi)^0.3 + correction(fR)) / (pi * M * MSUN_TO_SECONDS)
        fcut = (0.3236 + 0.04894 * chi + 0.01346 * chi^2 + correction(fC)) / (pi * M * MSUN_TO_SECONDS)
        sigma = (0.25 * (1 - chi)^0.45 - 0.1575 * (1 - chi)^0.75 + correction(sig)) / (pi * M * MSUN_TO_SECONDS)

        alpha = (0.0, -323 / 224 + 451 * eta / 168, (27 / 8 - 11 * eta / 6) * chi)
        eps = (1.4547 * chi - 1.8897, -1.8153 * chi + 1.6557, 0.0)
        vm = pn_velocity_powers(M, fmerge)
        vr = pn_velocity_powers(M, fring)
        wm = fmerge^(-1 / 3) * (1 + _dot3(alpha, vm))^2 / (fmerge^(2 / 3) * (1 + _dot3(eps, vm))^2 / fmerge)
        wr = (wm * fring^(2 / 3) * (1 + _dot3(eps, vr))^2 / fmerge) / (fring^2 / (fmerge * fring^(4 / 3)))

        for (i, fraw) in pairs(freqs)
            f = Float64(fraw)
            vf = pn_velocity_powers(M, f)
            if f < fmerge
                out[i] = f^(-1 / 3) * (1 + _dot3(alpha, vf))^2
            elseif f < fring
                out[i] = wm * f^(2 / 3) * (1 + _dot3(eps, vf))^2 / fmerge
            elseif f < fcut
                out[i] = wr * (f / (1 + ((f - fring) / (sigma / 2))^2))^2 / (fmerge * fring^(4 / 3))
            end
        end
    else
        fmerge = (0.29740 * eta^2 + 0.044810 * eta + 0.095560) / (pi * M * MSUN_TO_SECONDS)
        fring = (0.59411 * eta^2 + 0.089794 * eta + 0.19111) / (pi * M * MSUN_TO_SECONDS)
        fcut = (0.84845 * eta^2 + 0.12828 * eta + 0.27299) / (pi * M * MSUN_TO_SECONDS)
        sigma = (0.50801 * eta^2 + 0.077515 * eta + 0.022369) / (pi * M * MSUN_TO_SECONDS)
        for (i, fraw) in pairs(freqs)
            f = Float64(fraw)
            if f < fmerge
                out[i] = f^(-1 / 3)
            elseif f < fring
                out[i] = f^(2 / 3) / fmerge
            elseif f < fcut
                out[i] = (f / (1 + ((f - fring) / (sigma / 2))^2))^2 / (fmerge * fring^(4 / 3))
            end
        end
    end

    mchirp = eta^(3 / 5) * M * SI_M_SUN
    amp = (SI_G * pi)^(2 / 3) * mchirp^(5 / 3) / 3
    return amp .* out
end
dedf(Mtot::Real, freq::Real; kwargs...) = only(dedf(Mtot, [freq]; kwargs...))

"""
    precompute_omega_weights(rng, freqs; tmp_min=2, tmp_max=100, n=20000, ...)

Draw the same proposal variables as Python `precompute_omega_weights` and
return an `OmegaGWWeights` workspace. This function is deterministic for a
given Julia RNG.
"""
function precompute_omega_weights(rng::AbstractRNG, freqs; tmp_min=2.0, tmp_max=100.0, n::Integer=20000,
    chimax=nothing, inspiral_only::Bool=false, pn::Bool=true)
    freqs = Float64.(collect(freqs))
    m1s = rand(rng, n) .* (tmp_max - tmp_min) .+ tmp_min
    m2s = Vector{Float64}(undef, n)
    redshifts = rand(rng, n) .* 10.0
    p_m1 = fill(1 / (tmp_max - tmp_min), n)
    p_z = fill(0.1, n)
    p_m2 = Vector{Float64}(undef, n)
    dEdfs = Matrix{Float64}(undef, n, length(freqs))
    p_chi12 = chimax === nothing ? nothing : 1 / Float64(chimax)^2

    for i in 1:n
        m2s[i] = tmp_min + rand(rng) * (m1s[i] - tmp_min)
        p_m2[i] = 1 / (m1s[i] - tmp_min)
        q = m2s[i] / m1s[i]
        eta = q / (1 + q)^2
        chi = 0.0
        if chimax !== nothing
            chi1 = rand(rng) * Float64(chimax)
            chi2 = rand(rng) * Float64(chimax)
            delta_chi = (m1s[i] - m2s[i]) / (m1s[i] + m2s[i])
            chi = 0.5 * (1 + delta_chi) * chi1 + 0.5 * (1 - delta_chi) * chi2
        end
        dEdfs[i, :] = dedf(m1s[i] + m2s[i], freqs .* (1 + redshifts[i]); eta, inspiral_only, pn, chi)
    end

    return OmegaGWWeights(freqs, m1s, m2s, redshifts, p_m1, p_m2, p_z, dEdfs, p_chi12)
end
precompute_omega_weights(freqs; rng=Random.default_rng(), kwargs...) =
    precompute_omega_weights(rng, freqs; kwargs...)

_cosmo_h0_si(c::FlatLambdaCDM) = c.H0 * 1000 / (SI_KPC * 1e3)
_efunc(c::FlatLambdaCDM, z) = sqrt(c.Om0 * (1 + z)^3 + (1 - c.Om0))
_mass_pdf(mass::ConditionalMassDistribution, m1, m2) = pdf(mass, m1, m2)
_mass_pdf(mass, m1, m2) = applicable(pdf, mass, m1, m2) ? pdf(mass, m1, m2) : exp(logpdf(mass, m1, m2))

_omega_population(model) =
    throw(ArgumentError("spectral_siren_omega_gw supports vanilla FlatLambdaCDM CBC populations; got $(typeof(model))"))
_omega_population(model::SimplePowerLawPopulation) =
    (model.cosmology, model.mass, model.redshift_rate, model.R0)
_omega_population(model::CBCVanillaRate{<:FlatLambdaCDM}) =
    (model.cosmology, model.mass_distribution, model.redshift_rate, model.R0)
_omega_population(model::SpinWeightedRate) = _omega_population(model.base)
_omega_population(model::PEOnlySpinWeightedRate) = _omega_population(model.base)

"""
    spectral_siren_omega_gw(model, weights)

Compute the stochastic background spectrum for a vanilla CBC population and an
`OmegaGWWeights` workspace. This is the Julia-native equivalent of the Python
`spectral_siren_vanilla_omega_gw` path for vanilla CBC populations.
"""
function spectral_siren_omega_gw(model, weights::OmegaGWWeights; return_diagnostics::Bool=false)
    cosmology, mass_distribution, redshift_rate, R0 = _omega_population(model)
    H0 = _cosmo_h0_si(cosmology)
    rho_c = 3 * (H0 * SI_C)^2 / (8pi * SI_G) * (SI_KPC * 1e3)^3
    sample_weights = Vector{Float64}(undef, length(weights))
    for i in eachindex(sample_weights)
        pm = _mass_pdf(mass_distribution, weights.m1s[i], weights.m2s[i])
        pr = exp(log_rate(redshift_rate, weights.redshifts[i])) * R0
        rate = pr / _efunc(cosmology, weights.redshifts[i]) / (1 + weights.redshifts[i])
        sample_weights[i] = rate * pm / (weights.p_z[i] * weights.p_m1[i] * weights.p_m2[i])
    end
    omega = Vector{Float64}(undef, length(weights.frequencies))
    for j in eachindex(omega)
        accum = 0.0
        for i in eachindex(sample_weights)
            accum += weights.dEdfs[i, j] * sample_weights[i]
        end
        omega[j] = weights.frequencies[j] * (accum / length(weights)) / (rho_c * H0 * 1e9 * YEAR_SECONDS)
    end
    if return_diagnostics
        diag = StochasticDiagnostics(
            nweights=length(weights),
            nfrequencies=length(weights.frequencies),
            max_weight=maximum(sample_weights),
            min_weight=minimum(sample_weights),
            has_nan=any(isnan, sample_weights) || any(isnan, omega),
            has_inf=any(isinf, sample_weights) || any(isinf, omega),
        )
        return omega, diag
    end
    return omega
end

function _check_stochastic_axes(weights::OmegaGWWeights, data::StochasticData)
    length(weights.frequencies) == length(data.frequencies) ||
        throw(ArgumentError("OmegaGW weights and stochastic data have different frequency counts"))
    all(isapprox.(weights.frequencies, data.frequencies; rtol=1e-12, atol=1e-12)) ||
        throw(ArgumentError("OmegaGW weights and stochastic data frequencies do not match"))
    return nothing
end

"""
    stochastic_loglikelihood(model, weights, data)

Gaussian stochastic-background log-likelihood, matching Python
`Stochastic_likelihood_only` up to the omitted normalization constant.
"""
function stochastic_loglikelihood(model, weights::OmegaGWWeights, data::StochasticData)
    _check_stochastic_axes(weights, data)
    cosmology = first(_omega_population(model))
    hscale = cosmology.H0 / data.reference_H0
    cf = data.Cf .* hscale^(-2)
    sigma2s = data.sigma2s .* hscale^(-4)
    omega = spectral_siren_omega_gw(model, weights)
    diff = abs.(omega .- cf)
    value = -0.5 * sum((diff .^ 2) ./ sigma2s)
    return isfinite(value) ? value : -Inf
end

"""
    joint_loglikelihood(model, population_data, weights, stochastic_data; options=LikelihoodOptions())

Poisson CBC hierarchical likelihood plus the stochastic-background likelihood.
"""
function joint_loglikelihood(model, population_data::PopulationData,
    weights::OmegaGWWeights, stochastic_data::StochasticData; options::LikelihoodOptions=LikelihoodOptions())
    cbc = loglikelihood(model, population_data; options)
    isfinite(cbc) || return -Inf
    return cbc + stochastic_loglikelihood(model, weights, stochastic_data)
end

function stochastic_planned()
    throw(ErrorException("Catalog/EM stochastic joint likelihoods are future API design; stochastic-only and vanilla CBC+stochastic likelihood helpers are available."))
end

end

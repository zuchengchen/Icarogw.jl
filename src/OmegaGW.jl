module OmegaGW

using ..Stochastic

export PNVelocityPowers,
    OmegaGWWeights,
    StochasticDiagnostics,
    pn_velocity_powers,
    dedf,
    precompute_omega_weights,
    spectral_siren_omega_gw,
    omega_gw_planned

const PNVelocityPowers = Stochastic.PNVelocityPowers
const OmegaGWWeights = Stochastic.OmegaGWWeights
const StochasticDiagnostics = Stochastic.StochasticDiagnostics
const pn_velocity_powers = Stochastic.pn_velocity_powers
const dedf = Stochastic.dedf
const precompute_omega_weights = Stochastic.precompute_omega_weights
const spectral_siren_omega_gw = Stochastic.spectral_siren_omega_gw

function omega_gw_planned()
    throw(ErrorException("Stochastic-only likelihoods and catalog/stochastic mixed likelihoods are not implemented yet. The Omega_GW energy-spectrum and vanilla spectral-siren helpers are available."))
end

end

module OmegaGW

using ..Stochastic

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
    omega_gw_planned

const PNVelocityPowers = Stochastic.PNVelocityPowers
const OmegaGWWeights = Stochastic.OmegaGWWeights
const StochasticData = Stochastic.StochasticData
const StochasticDiagnostics = Stochastic.StochasticDiagnostics
const pn_velocity_powers = Stochastic.pn_velocity_powers
const dedf = Stochastic.dedf
const precompute_omega_weights = Stochastic.precompute_omega_weights
const read_stochastic_csv = Stochastic.read_stochastic_csv
const write_stochastic_hdf5 = Stochastic.write_stochastic_hdf5
const read_stochastic_hdf5 = Stochastic.read_stochastic_hdf5
const spectral_siren_omega_gw = Stochastic.spectral_siren_omega_gw
const stochastic_loglikelihood = Stochastic.stochastic_loglikelihood
const joint_loglikelihood = Stochastic.joint_loglikelihood

function omega_gw_planned()
    throw(ErrorException("Catalog/stochastic mixed likelihoods with catalog/EM rate models are not implemented yet. The Omega_GW energy-spectrum, vanilla spectral-siren, stochastic-only, and vanilla CBC+stochastic helpers are available."))
end

end

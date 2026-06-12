module OmegaGW

export omega_gw_planned

"""
    omega_gw_planned()

Omega_GW spectral-siren helpers are planned for a later migration phase and are
intentionally not implemented in the first native Julia version.
"""
function omega_gw_planned()
    throw(ErrorException("Omega_GW functionality is planned, not implemented in Icarogw.jl first-version scope."))
end

end

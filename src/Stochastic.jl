module Stochastic

export stochastic_planned

"""
    stochastic_planned()

Stochastic-background likelihoods are planned for a later migration phase and
are intentionally not implemented in the first native Julia version.
"""
function stochastic_planned()
    throw(ErrorException("Stochastic-background functionality is planned, not implemented in Icarogw.jl first-version scope."))
end

end

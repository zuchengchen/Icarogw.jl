module Catalog

export catalog_planned

"""
    catalog_planned()

Galaxy catalog, dark-siren, and bright-siren functionality is planned but not
implemented in the first native Julia version. This placeholder exists so users
get an explicit error instead of a silent partial implementation.
"""
function catalog_planned()
    throw(ErrorException("Catalog and EM-counterpart functionality is planned, not implemented in Icarogw.jl first-version scope."))
end

end

module Conversions

using ..Cosmology

export chirp_mass,
    mass_ratio,
    f_gw_isco,
    L2M,
    M2L,
    apparent_magnitude,
    absolute_magnitude,
    source_to_detector,
    detector_to_source,
    source2detector,
    detector2source,
    detector_to_source_jacobian,
    detector_to_source_jacobian_q,
    detector_to_source_jacobian_single_mass,
    source_to_detector_jacobian,
    chi_eff_from_spins,
    chi_p_from_spins,
    cartesian_spins_to_chis

"""
    chirp_mass(m1, m2)

Binary chirp mass in solar masses:
`(m1*m2)^(3/5) / (m1 + m2)^(1/5)`.
"""
chirp_mass(m1::Real, m2::Real) = (m1 * m2)^(3 / 5) / (m1 + m2)^(1 / 5)
chirp_mass(m1::AbstractArray, m2::AbstractArray) = chirp_mass.(m1, m2)

"""
    mass_ratio(m1, m2)

Mass ratio `q = m2 / m1`. The convention assumes `m1 >= m2`, so valid
population-model values are normally in `(0, 1]`.
"""
mass_ratio(m1, m2) = m2 / m1

"""
    f_gw_isco(m1, m2)

Approximate gravitational-wave ISCO frequency in Hz for detector-frame masses
in solar masses. This follows the Python helper `f_GW_ISCO`.
"""
f_gw_isco(m1, m2) = 2 * (2.20 * 10^3 / (m1 + m2))

"""
    L2M(L)

Convert luminosity in watt to absolute bolometric magnitude using the IAU 2015
zero point used by the Python project.
"""
L2M(L) = -2.5 * log10(L) + 71.197425

"""
    M2L(M)

Convert absolute bolometric magnitude to luminosity in watt.
"""
M2L(M) = 3.0128e28 * 10.0^(-0.4 * M)

"""
    apparent_magnitude(M, dl, kcorr=0)

Convert absolute magnitude `M` to apparent magnitude for luminosity distance
`dl` in Mpc and K-correction `kcorr`.
"""
apparent_magnitude(M, dl, kcorr=0) = M + 5log10(dl) + 25 + kcorr

"""
    absolute_magnitude(m, dl, kcorr=0)

Convert apparent magnitude `m` to absolute magnitude for luminosity distance
`dl` in Mpc and K-correction `kcorr`.
"""
absolute_magnitude(m, dl, kcorr=0) = m - 5log10(dl) - 25 - kcorr

"""
    source_to_detector(m1_source, m2_source, z, cosmology)

Convert source-frame component masses to detector-frame masses and luminosity
distance. Returns `(m1_detector, m2_detector, luminosity_distance_Mpc)`.
"""
function source_to_detector(m1_source, m2_source, z, cosmology::AbstractCosmology)
    return m1_source .* (1 .+ z), m2_source .* (1 .+ z), luminosity_distance(cosmology, z)
end
source2detector(args...) = source_to_detector(args...)

"""
    detector_to_source(m1_detector, m2_detector, dl, cosmology)

Convert detector-frame masses and luminosity distance in Mpc to source-frame
masses and redshift. Returns `(m1_source, m2_source, z)`.
"""
function detector_to_source(m1_detector, m2_detector, dl, cosmology::AbstractCosmology)
    z = redshift_at_luminosity_distance(cosmology, dl)
    return m1_detector ./ (1 .+ z), m2_detector ./ (1 .+ z), z
end
detector2source(args...) = detector_to_source(args...)

"""
    detector_to_source_jacobian(z, cosmology)

Jacobian `|d(m1d,m2d,dL) / d(m1s,m2s,z)| = (1+z)^2 ddL/dz`.
"""
detector_to_source_jacobian(z, cosmology::AbstractCosmology) = abs((1 + z)^2 * ddl_dz(cosmology, z))

"""
    detector_to_source_jacobian_q(z, cosmology)

Jacobian `|d(m1d,q,dL) / d(m1s,q,z)| = (1+z) ddL/dz`.
"""
detector_to_source_jacobian_q(z, cosmology::AbstractCosmology) = abs((1 + z) * ddl_dz(cosmology, z))

"""
    detector_to_source_jacobian_single_mass(z, cosmology)

Jacobian `|d(md,dL) / d(ms,z)| = (1+z) ddL/dz`.
"""
detector_to_source_jacobian_single_mass(z, cosmology::AbstractCosmology) =
    detector_to_source_jacobian_q(z, cosmology)

"""
    source_to_detector_jacobian(z, cosmology)

Inverse of `detector_to_source_jacobian`.
"""
source_to_detector_jacobian(z, cosmology::AbstractCosmology) =
    inv(detector_to_source_jacobian(z, cosmology))

"""
    chi_eff_from_spins(chi1, chi2, cos1, cos2, q)

Effective aligned spin `(chi1*cos1 + q*chi2*cos2) / (1 + q)`.
"""
chi_eff_from_spins(chi1, chi2, cos1, cos2, q) = (chi1 * cos1 + q * chi2 * cos2) / (1 + q)

"""
    chi_p_from_spins(chi1, chi2, cos1, cos2, q)

Approximate precessing-spin parameter following the Python helper.
"""
function chi_p_from_spins(chi1, chi2, cos1, cos2, q)
    s1p = chi1 * sqrt(max(0, 1 - cos1^2))
    s2p = chi2 * sqrt(max(0, 1 - cos2^2))
    return max(s1p, q * (4q + 3) / (4 + 3q) * s2p)
end

"""
    cartesian_spins_to_chis(s1x, s1y, s1z, s2x, s2y, s2z, q)

Convert Cartesian dimensionless spin components to `(chi_eff, chi_p)`.
"""
function cartesian_spins_to_chis(s1x, s1y, s1z, s2x, s2y, s2z, q)
    chi1 = sqrt(s1x^2 + s1y^2 + s1z^2)
    chi2 = sqrt(s2x^2 + s2y^2 + s2z^2)
    cos1 = chi1 == 0 ? 1.0 : s1z / chi1
    cos2 = chi2 == 0 ? 1.0 : s2z / chi2
    return chi_eff_from_spins(chi1, chi2, cos1, cos2, q),
        chi_p_from_spins(chi1, chi2, cos1, cos2, q)
end

end

#!/usr/bin/env python3
"""Generate small reference fixtures from the local Python ``../icarogw``.

This script is a development tool. The Julia package must not import Python at
runtime; generated fixture files are static test inputs committed under
``test/reference`` when they are small and license-safe.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import math
import pathlib
import sys
import types
import numpy as np
import scipy.integrate
from typing import Iterable


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_REF = (REPO_ROOT / ".." / "icarogw").resolve()
DEFAULT_OUT = REPO_ROOT / "test" / "reference"


def _prepend_python_reference() -> None:
    sys.path.insert(0, str(PYTHON_REF))


def _load_reference_module(module_name: str, relative_path: str):
    """Load one Python reference file without importing package ``__init__``.

    The Python package imports catalog/skymap dependencies at package import
    time. Lightweight fixture suites such as stochastic ``dEdf`` should not
    require the full catalog stack, so they load their source file directly.
    """

    path = PYTHON_REF / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_lightweight_reference_package() -> None:
    """Install a minimal ``icarogw`` package for direct file imports.

    Some reference files use relative imports but the package ``__init__``
    imports optional catalog/skymap dependencies. This helper gives direct
    module loads enough package context without triggering the heavy import.
    """

    package = sys.modules.get("icarogw")
    if package is None:
        package = types.ModuleType("icarogw")
        package.__path__ = [str(PYTHON_REF / "icarogw")]
        sys.modules["icarogw"] = package
    cupy_pal = _load_reference_module("icarogw.cupy_pal", "icarogw/cupy_pal.py")
    sys.modules["icarogw.cupy_pal"] = cupy_pal
    package.cupy_pal = cupy_pal

    cosmology_stub = types.ModuleType("icarogw.cosmology")

    class _UnavailableAstropyCosmology:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("cosmology-dependent simulation fixtures are not part of this lightweight suite")

    cosmology_stub.astropycosmology = _UnavailableAstropyCosmology
    cosmology_stub.alphalog_astropycosmology = _UnavailableAstropyCosmology
    cosmology_stub.cM_astropycosmology = _UnavailableAstropyCosmology
    cosmology_stub.extraD_astropycosmology = _UnavailableAstropyCosmology
    cosmology_stub.Xi0_astropycosmology = _UnavailableAstropyCosmology
    cosmology_stub.eps0_astropycosmology = _UnavailableAstropyCosmology
    cosmology_stub.md_rate = _UnavailableAstropyCosmology
    cosmology_stub.md_gamma_rate = _UnavailableAstropyCosmology
    cosmology_stub.powerlaw_rate = _UnavailableAstropyCosmology
    cosmology_stub.beta_rate = _UnavailableAstropyCosmology
    cosmology_stub.beta_rate_line = _UnavailableAstropyCosmology
    sys.modules.setdefault("icarogw.cosmology", cosmology_stub)
    package.cosmology = sys.modules["icarogw.cosmology"]

    astropy_stub = types.ModuleType("astropy")
    astropy_cosmology_stub = types.ModuleType("astropy.cosmology")
    astropy_cosmology_stub.FlatLambdaCDM = _UnavailableAstropyCosmology
    astropy_cosmology_stub.FlatwCDM = _UnavailableAstropyCosmology
    astropy_cosmology_stub.Flatw0waCDM = _UnavailableAstropyCosmology
    astropy_stub.cosmology = astropy_cosmology_stub
    sys.modules.setdefault("astropy", astropy_stub)
    sys.modules.setdefault("astropy.cosmology", astropy_cosmology_stub)

    conversions_stub = types.ModuleType("icarogw.conversions")
    conversions_stub.L2M = lambda luminosity: -2.5 * math.log10(luminosity) + 71.197425
    conversions_stub.M2L = lambda magnitude: 3.0128e28 * 10.0 ** (-0.4 * magnitude)
    sys.modules.setdefault("icarogw.conversions", conversions_stub)
    package.conversions = sys.modules["icarogw.conversions"]

    wrappers_stub = types.ModuleType("icarogw.wrappers")
    wrappers_stub.__all__ = []
    wrappers_stub.massprior_PowerLawPeak = object
    wrappers_stub.m1m2_conditioned = lambda value: value
    sys.modules.setdefault("icarogw.wrappers", wrappers_stub)
    package.wrappers = sys.modules["icarogw.wrappers"]

    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable=None, *args, **kwargs: iterable if iterable is not None else []
    sys.modules.setdefault("tqdm", tqdm_stub)


def _load_lightweight_conversions_module():
    """Load ``conversions.py`` with catalog/skymap imports stubbed out."""

    _install_lightweight_reference_package()

    healpy_stub = types.ModuleType("healpy")
    sys.modules.setdefault("healpy", healpy_stub)

    ligo_stub = types.ModuleType("ligo")
    skymap_stub = types.ModuleType("ligo.skymap")
    skymap_io_stub = types.ModuleType("ligo.skymap.io")
    skymap_fits_stub = types.ModuleType("ligo.skymap.io.fits")
    skymap_fits_stub.read_sky_map = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("skymap fixtures are not part of this lightweight suite")
    )
    ligo_stub.skymap = skymap_stub
    skymap_stub.io = skymap_io_stub
    skymap_io_stub.fits = skymap_fits_stub
    sys.modules.setdefault("ligo", ligo_stub)
    sys.modules.setdefault("ligo.skymap", skymap_stub)
    sys.modules.setdefault("ligo.skymap.io", skymap_io_stub)
    sys.modules.setdefault("ligo.skymap.io.fits", skymap_fits_stub)

    astropy_healpix_stub = types.ModuleType("astropy_healpix")
    sys.modules.setdefault("astropy_healpix", astropy_healpix_stub)

    astropy_stub = sys.modules.setdefault("astropy", types.ModuleType("astropy"))
    units_stub = types.ModuleType("astropy.units")
    units_stub.rad = 1.0
    astropy_stub.units = units_stub
    sys.modules.setdefault("astropy.units", units_stub)

    return _load_reference_module("icarogw.reference_conversions", "icarogw/conversions.py")


def _install_mpmath_stub() -> None:
    """Install the tiny ``mpmath.gammainc`` surface used by ``cosmology.py``."""

    if "mpmath" in sys.modules:
        return
    mpmath_stub = types.ModuleType("mpmath")

    def gammainc(shape, a=None, b=None):
        lo = 0.0 if a is None else float(a)
        hi = math.inf if b is None else float(b)
        value, _ = scipy.integrate.quad(
            lambda x: x ** (float(shape) - 1) * math.exp(-x),
            lo,
            hi,
            epsabs=1e-12,
            epsrel=1e-12,
            limit=200,
        )
        return value

    mpmath_stub.gammainc = gammainc
    sys.modules["mpmath"] = mpmath_stub


def _load_lightweight_cosmology_module():
    """Load ``cosmology.py`` without requiring the full Python catalog stack."""

    _install_lightweight_reference_package()
    _install_mpmath_stub()
    return _load_reference_module("icarogw.reference_cosmology", "icarogw/cosmology.py")


def _write_rows(path: pathlib.Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate_core(outdir: pathlib.Path) -> list[pathlib.Path]:
    """Regenerate lightweight conversion fixtures.

    The existing repository already contains broader core fixtures. This suite
    is intentionally tiny and dependency-light so contributors can verify their
    Python reference environment before adding heavier catalog/stochastic
    fixture suites.
    """

    conversions = _load_lightweight_conversions_module()
    m1 = 36.0
    m2 = 21.0
    luminosity = 3.0128e28
    magnitude = -19.2
    chi1 = 0.4
    chi2 = 0.7
    cos1 = 0.3
    cos2 = -0.2
    q = m2 / m1
    q_spin = np.array([0.45, 0.80])
    xeff_spin = np.array([0.10, -0.20])
    xp_spin = np.array([0.25, 0.40])
    np.random.seed(321)
    conditional_chi_p_prior = conversions.chi_p_prior_given_chi_eff_q(
        q_spin[0], 1.0, xeff_spin[0], xp_spin[0], ndraws=2000
    )
    np.random.seed(322)
    joint_spin_prior = conversions.joint_prior_from_isotropic_spins(
        q_spin, 1.0, xeff_spin, xp_spin, ndraws=2000
    )

    rows = [
        {
            "m1": m1,
            "m2": m2,
            "chirp_mass": (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5),
            "mass_ratio": m2 / m1,
            "f_gw_isco": 2 * (2.20 * (1 / (m1 + m2))) * 10**3,
            "L": luminosity,
            "L2M": -2.5 * math.log10(luminosity) + 71.197425,
            "M": magnitude,
            "M2L": 3.0128e28 * 10.0 ** (-0.4 * magnitude),
            "chi1": chi1,
            "chi2": chi2,
            "cos1": cos1,
            "cos2": cos2,
            "q": q,
            "chi_eff": (chi1 * cos1 + q * chi2 * cos2) / (1 + q),
            "chi_p": max(chi1 * (1 - cos1**2) ** 0.5, q * (4 * q + 3) / (4 + 3 * q) * chi2 * (1 - cos2**2) ** 0.5),
            "q_spin_1": q_spin[0],
            "xeff_spin_1": xeff_spin[0],
            "xp_spin_1": xp_spin[0],
            "conditional_chi_p_prior": conditional_chi_p_prior,
            "q_spin_2": q_spin[1],
            "xeff_spin_2": xeff_spin[1],
            "xp_spin_2": xp_spin[1],
            "joint_spin_prior_1": joint_spin_prior[0],
            "joint_spin_prior_2": joint_spin_prior[1],
        }
    ]
    path = outdir / "generated_reference_conversions_core.csv"
    _write_rows(
        path,
        (
            "m1",
            "m2",
            "chirp_mass",
            "mass_ratio",
            "f_gw_isco",
            "L",
            "L2M",
            "M",
            "M2L",
            "chi1",
            "chi2",
            "cos1",
            "cos2",
            "q",
            "chi_eff",
            "chi_p",
            "q_spin_1",
            "xeff_spin_1",
            "xp_spin_1",
            "conditional_chi_p_prior",
            "q_spin_2",
            "xeff_spin_2",
            "xp_spin_2",
            "joint_spin_prior_1",
            "joint_spin_prior_2",
        ),
        rows,
    )
    return [path]


def generate_stochastic_smoke(outdir: pathlib.Path) -> list[pathlib.Path]:
    """Generate a tiny stochastic spectrum fixture.

    This suite intentionally avoids random omega-weight tables. It captures the
    deterministic ``dEdf`` formula at a few frequencies and is the first
    fixture to use when implementing the Julia stochastic/OmegaGW module.
    """

    omega_gw = _load_reference_module("icarogw_reference_omega_gw", "icarogw/omega_gw.py")

    freqs = [10.0, 25.0, 75.0, 150.0]
    values = omega_gw.dEdf(60.0, omega_gw.np.array(freqs), eta=0.24, PN=True, chi=0.2)
    rows = [{"Mtot": 60.0, "eta": 0.24, "chi": 0.2, "freq": f, "dEdf": v} for f, v in zip(freqs, values)]
    path = outdir / "generated_reference_stochastic_dedf.csv"
    _write_rows(path, ("Mtot", "eta", "chi", "freq", "dEdf"), rows)
    return [path]


def generate_simulation_utils_smoke(outdir: pathlib.Path) -> list[pathlib.Path]:
    """Generate deterministic simulation and utility formula fixtures."""

    _install_lightweight_reference_package()
    simulation = _load_reference_module("icarogw.reference_simulation", "icarogw/simulation.py")
    cupy_pal = sys.modules["icarogw.cupy_pal"]
    stats = simulation.stats
    np = simulation.np

    m1 = np.array([30.0, 36.0, 50.0])
    m2 = np.array([20.0, 18.0, 25.0])
    z = np.array([0.10, 0.20, 0.35])
    snr = np.array([13.0, 9.0, 20.0])
    theta = np.array([0.5, 1.0, 1.2])
    d_l = np.array([500.0, 1000.0, 1500.0])
    m1d = m1 * (1 + z)
    m2d = m2 * (1 + z)
    rho_true_det = 9.0 * theta * np.power(simulation.chirp_mass(m1d, m2d) / 25.0, 5 / 6) * ((1.5 * 1000) / d_l)
    qs = np.array([0.60, 0.50, 0.80])
    mds = np.array([24.0, 30.0, 42.0])
    thetas = np.array([0.5, 0.8, 1.1])
    rho_model = np.array([10.0, 14.0, 18.0])
    rho_obs = np.array([11.0, 13.0, 19.0])
    q_obs = np.array([0.62, 0.48, 0.78])
    md_obs = np.array([24.1, 29.8, 42.5])
    theta_obs = np.array([0.55, 0.75, 1.05])
    likelihood = (
        stats.ncx2.pdf(rho_obs**2.0, 6, np.power(rho_model, 2.0))
        * stats.norm.pdf(q_obs, qs, 0.25 * qs * 10 / rho_obs)
        * stats.norm.pdf(md_obs, mds, 10 ** (-3) * mds * 10 / rho_obs)
        * stats.norm.pdf(theta_obs, thetas, 0.3 * 10 / rho_obs)
    )

    rows = []
    bounds_1d = cupy_pal.check_bounds_1D(np.array([-1.0, 0.5, 2.0]), 0.0, 1.0)
    bounds_2d = cupy_pal.check_bounds_2D(np.array([1.0, 0.5, 2.0]), np.array([0.5, 0.7, 1.0]), np.array([0.1, np.nan, 0.3]))
    idx_snr_freq = np.where((snr >= 12.0) & (simulation.f_GW(m1, m2, z) > 15.0))[0]
    idx_snr_flat = np.where(snr >= 12.0)[0]
    for i in range(len(m1)):
        rows.append(
            {
                "i": i + 1,
                "m1": m1[i],
                "m2": m2[i],
                "z": z[i],
                "chirp_mass": simulation.chirp_mass(m1[i], m2[i]),
                "chirp_mass_det": simulation.chirp_mass_det(m1[i], m2[i], z[i]),
                "mass_ratio": simulation.mass_ratio(m1[i], m2[i]),
                "f_gw": simulation.f_GW(m1[i], m2[i], z[i]),
                "snr_flat_alpha2": simulation.snr_samples_flat(z[i], alpha=2.0),
                "rho_true_detector": rho_true_det[i],
                "likelihood": likelihood[i],
                "bounds_1d": bool(bounds_1d[i]),
                "bounds_2d": bool(bounds_2d[i]),
                "passes_snr_freq": i in idx_snr_freq,
                "passes_snr_flat": i in idx_snr_flat,
            }
        )

    path = outdir / "generated_reference_simulation_utils.csv"
    _write_rows(
        path,
        (
            "i",
            "m1",
            "m2",
            "z",
            "chirp_mass",
            "chirp_mass_det",
            "mass_ratio",
            "f_gw",
            "snr_flat_alpha2",
            "rho_true_detector",
            "likelihood",
            "bounds_1d",
            "bounds_2d",
            "passes_snr_freq",
            "passes_snr_flat",
        ),
        rows,
    )
    return [path]


def generate_priors_rates_smoke(outdir: pathlib.Path) -> list[pathlib.Path]:
    """Generate deterministic priors/rates parity fixtures."""

    _install_lightweight_reference_package()
    _prepend_python_reference()
    import scipy.special  # noqa: F401 - ensures priors.py can access scipy.special
    import scipy.stats  # noqa: F401 - wrappers access scipy.stats through the scipy module

    priors = _load_reference_module("icarogw.reference_priors", "icarogw/priors.py")
    sys.modules["icarogw.priors"] = priors
    sys.modules["icarogw"].priors = priors
    wrappers = _load_reference_module("icarogw.reference_wrappers", "icarogw/wrappers.py")
    np = priors.np

    xs = np.array([4.0, 5.5, 7.0])
    bivar = priors.Bivariate2DGaussian(
        x1min=-1.0,
        x1max=1.0,
        x1mean=0.0,
        x2min=-2.0,
        x2max=2.0,
        x2mean=0.1,
        x1variance=0.5,
        x12covariance=0.1,
        x2variance=1.0,
    )
    x1 = np.array([-0.5, 0.1, 1.5])
    x2 = np.array([0.0, 0.2, 0.4])
    log_rate_1 = np.array([-10.0, -11.0, -12.5])
    log_rate_2 = np.array([-9.5, -12.0, -12.0])
    lambda_pop = 0.35
    mixed_rate = np.logaddexp(np.log(lambda_pop) + log_rate_1, np.log1p(-lambda_pop) + log_rate_2)
    mix_mass = np.array([8.0, 20.0, 35.0])
    mix_redshift = np.array([0.0, 0.5, 1.0])
    m1_pair = np.array([12.0, 20.0, 35.0])
    m2_pair = np.array([8.0, 12.0, 6.0])
    chi1 = np.array([0.2, 0.45, 0.7])
    chi2 = np.array([0.3, 0.35, 0.6])
    cos1 = np.array([0.4, 0.1, -0.2])
    cos2 = np.array([-0.2, 0.3, 0.7])
    mass1_source = np.array([20.0, 35.0, 50.0])
    mass2_source = np.array([15.0, 25.0, 30.0])
    domega220 = np.array([-0.5, 0.0, 0.7])
    dtau220 = np.array([0.2, -0.3, 0.5])

    rows = []
    high = priors._highpass_filter(xs, 5.0, 2.0)
    low = priors._lowpass_filter(xs, 8.0, 2.0)
    notch = priors._notch_filter(xs, 5.0, 2.0, 8.0, 2.0, 0.4)
    mixed_linear = priors._mixed_linear_function(xs / 10.0, 0.2, 0.8)
    mixed_sigmoid = priors._mixed_double_sigmoid_function(xs / 10.0, 0.55, 0.15, 0.2, 0.8)
    lowpass_evolving = priors.LowpassSmoothedProbEvolving(priors.PowerLaw(5.0, 12.0, -2.0), 2.0)
    lowpass_evolving_logpdf = lowpass_evolving.log_pdf(xs)
    M_abs = np.array([-22.0, -19.5, -17.0])
    abs_l_powerlaw = priors.absL_PL_inM(-23.0, -16.0, -1.1)
    abs_l_powerlaw_logpdf = abs_l_powerlaw.log_pdf(M_abs)
    abs_l_powerlaw_cdf = abs_l_powerlaw.cdf(M_abs)
    log_bivar = bivar.log_pdf(x1, x2)

    spin_gaussian = wrappers.spinprior_default_gaussian()
    spin_gaussian.update(mu_chi_1=0.25, mu_chi_2=0.35, sigma_chi_1=0.2, sigma_chi_2=0.25, sigma_t=0.5, csi_spin=0.4)
    spin_evolving = wrappers.spinprior_default_evolving_gaussian()
    spin_evolving.update(mu_chi=0.15, sigma_chi=0.2, mu_dot=0.002, sigma_dot=0.001, sigma_t=0.6, csi_spin=0.3)
    spin_beta_gauss = wrappers.spinprior_default_beta_window_gaussian()
    spin_beta_gauss.update(mt=0.8, delta_mt=0.2, mix_f=40.0, alpha_chi=2.0, beta_chi=3.0,
                           mu_chi=0.35, sigma_chi=0.2, sigma_t=0.5, csi_spin=0.4)
    spin_beta_beta = wrappers.spinprior_default_beta_window_beta()
    spin_beta_beta.update(mt=0.8, delta_mt=0.2, mix_f=40.0, alpha_chi_low=2.0, beta_chi_low=3.0,
                          alpha_chi_high=4.0, beta_chi_high=2.5, sigma_t=0.5, csi_spin=0.4)
    pseob = wrappers.pseobprior_gaussian()
    pseob.update(mu_domega220=0.1, sigma_domega220=1.2, mu_dtau220=-0.2, sigma_dtau220=1.5, rho_pseob=0.25)
    eco = wrappers.spinprior_ECOs_totally_reflective()
    eco.update(alpha_chi=2.0, beta_chi=3.0, eps=1e-10, f_eco=0.35, sigma_chi_ECO=0.03)

    spin_gaussian_logpdf = spin_gaussian.log_pdf(chi1, chi2, cos1, cos2)
    spin_evolving_logpdf = spin_evolving.log_pdf(chi1, chi2, cos1, cos2, mass1_source, mass2_source)
    spin_beta_gauss_logpdf = spin_beta_gauss.log_pdf(chi1, chi2, cos1, cos2, mass1_source, mass2_source)
    spin_beta_beta_logpdf = spin_beta_beta.log_pdf(chi1, chi2, cos1, cos2, mass1_source, mass2_source)
    pseob_logpdf = pseob.log_pdf(domega220, dtau220)
    eco_logpdf = eco.log_pdf(chi1, chi2)

    pl_pl = wrappers.PowerLaw_PowerLaw(flag_powerlaw_smoothing=0)
    pl_pl.update(alpha_a=1.5, mmin_a=5.0, mmax_a=50.0,
                 alpha_b=2.5, mmin_b=8.0, mmax_b=80.0, mix=0.65)
    pl_pl_logpdf = pl_pl.log_pdf(mix_mass)

    pl_pl_g = wrappers.PowerLaw_PowerLaw_Gaussian(flag_powerlaw_smoothing=0)
    pl_pl_g.update(alpha_a=1.5, mmin_a=5.0, mmax_a=50.0,
                   alpha_b=2.5, mmin_b=8.0, mmax_b=80.0,
                   mu_g=30.0, sigma_g=4.0, mix_alpha=0.45, mix_beta=0.35)
    pl_pl_g_logpdf = pl_pl_g.log_pdf(mix_mass)

    pl_g_z = wrappers.PowerLawRedshiftLinear_GaussianRedshiftLinear(flag_powerlaw_smoothing=0)
    pl_g_z.update(alpha_z0=1.5, alpha_z1=0.2, mmin_z0=5.0, mmin_z1=0.5, mmax_z0=60.0, mmax_z1=1.0,
                  mu_z0=25.0, mu_z1=2.0, sigma_z0=4.0, sigma_z1=0.5,
                  mix_z0=0.75, mix_z1=0.35)
    pl_g_z_logpdf = pl_g_z.log_pdf(mix_mass, mix_redshift)

    g_g_z = wrappers.GaussianRedshiftLinear_GaussianRedshiftLinear()
    g_g_z.update(mu_a_z0=20.0, mu_a_z1=1.0, sigma_a_z0=3.0, sigma_a_z1=0.4,
                 mu_b_z0=40.0, mu_b_z1=-2.0, sigma_b_z0=5.0, sigma_b_z1=0.2,
                 mix_z0=0.6, mix_z1=0.3, mmin_g=5.0)
    g_g_z_logpdf = g_g_z.log_pdf(mix_mass, mix_redshift)

    np.random.seed(123)
    wrapper_powerlaw = wrappers.massprior_PowerLaw()
    paired_dip = wrappers.m1m2_paired_massratio_dip(wrapper_powerlaw)
    paired_dip.update(alpha=2.0, mmin=5.0, mmax=60.0, beta=1.2,
                      bottomsmooth=2.0, topsmooth=5.0, leftdip=10.0, rightdip=20.0,
                      leftdipsmooth=2.0, rightdipsmooth=3.0, deep=0.4)
    paired_dip_logpdf = paired_dip.log_pdf(m1_pair, m2_pair)

    np.random.seed(124)
    wrapper_powerlaw_general = wrappers.massprior_PowerLaw()
    paired_dip_general = wrappers.m1m2_paired_massratio_dip_general(wrapper_powerlaw_general)
    paired_dip_general.update(alpha=2.0, mmin=5.0, mmax=60.0, beta_bottom=0.5, beta_top=2.0,
                              bottomsmooth=2.0, topsmooth=5.0, leftdip=10.0, rightdip=20.0,
                              leftdipsmooth=2.0, rightdipsmooth=3.0, deep=0.4)
    paired_dip_general_logpdf = paired_dip_general.log_pdf(m1_pair, m2_pair)

    np.random.seed(125)
    paired_farah = wrappers.m1m2_paired_massratio_bpl_dip_farah_2022()
    paired_farah.update(alpha_1=1.5, alpha_2=3.0, mmin=5.0, mmax=60.0, beta_bottom=0.5, beta_top=2.0,
                        bottomsmooth=2.0, topsmooth=5.0, leftdip=12.0, rightdip=24.0,
                        leftdipsmooth=2.0, rightdipsmooth=3.0, deep=0.4)
    paired_farah_logpdf = paired_farah.log_pdf(m1_pair, m2_pair)

    np.random.seed(126)
    paired_bplmulti = wrappers.m1m2_paired_massratio_bplmulti_dip()
    paired_bplmulti.update(alpha_1=1.5, alpha_2=3.0, mmin=5.0, mmax=60.0, beta_bottom=0.5, beta_top=2.0,
                           bottomsmooth=2.0, topsmooth=5.0, leftdip=12.0, rightdip=24.0,
                           leftdipsmooth=2.0, rightdipsmooth=3.0, deep=0.4,
                           mu_g_low=10.0, sigma_g_low=1.5, lambda_g_low=0.4,
                           mu_g_high=35.0, sigma_g_high=3.0, lambda_g=0.15)
    paired_bplmulti_logpdf = paired_bplmulti.log_pdf(m1_pair, m2_pair)

    np.random.seed(127)
    paired_triple = wrappers.m1m2_paired_bpl_triplepeak_dip()
    paired_triple.update(alpha_1=1.5, alpha_2=3.0, mmin=5.0, mmax=60.0, beta_bottom=0.5, beta_top=2.0,
                         bottomsmooth=2.0, topsmooth=5.0, leftdip=12.0, rightdip=24.0,
                         leftdipsmooth=2.0, rightdipsmooth=3.0, deep=0.4,
                         mu_g_1=9.0, sigma_g_1=1.0, lambda_g=0.2,
                         mu_g_2=25.0, sigma_g_2=2.0, lambda_1=0.3,
                         mu_g_3=40.0, sigma_g_3=3.0, lambda_2=1.0)
    paired_triple_logpdf = paired_triple.log_pdf(m1_pair, m2_pair)

    paired_bplmulti_conditioned = wrappers.m1m2_paired_massratio_bplmulti_dip_conditioned()
    paired_bplmulti_conditioned.update(alpha_1=1.5, alpha_2=3.0, mmin=5.0, mmax=60.0,
                                       beta_bottom=0.5, beta_top=2.0,
                                       bottomsmooth=2.0, topsmooth=5.0, leftdip=12.0, rightdip=24.0,
                                       leftdipsmooth=2.0, rightdipsmooth=3.0, deep=0.4,
                                       mu_g_low=10.0, sigma_g_low=1.5, lambda_g_low=0.4,
                                       mu_g_high=35.0, sigma_g_high=3.0, lambda_g=0.15)
    paired_bplmulti_conditioned_logpdf = paired_bplmulti_conditioned.log_pdf(m1_pair, m2_pair)

    bin_model = wrappers.massprior_BinModel2d(2)
    bin_model.update(mmin=5.0, mmax=45.0, bin_0=1.0, bin_1=2.0, bin_2=3.0)
    bin_model_logpdf = bin_model.log_pdf(m1_pair, m2_pair)
    for i in range(len(xs)):
        rows.append(
            {
                "i": i + 1,
                "x": xs[i],
                "highpass": high[i],
                "lowpass": low[i],
                "notch": notch[i],
                "mixed_linear": mixed_linear[i],
                "mixed_sigmoid": mixed_sigmoid[i],
                "lowpass_evolving_logpdf": lowpass_evolving_logpdf[i],
                "M_abs": M_abs[i],
                "abs_l_powerlaw_logpdf": abs_l_powerlaw_logpdf[i],
                "abs_l_powerlaw_cdf": abs_l_powerlaw_cdf[i],
                "x1": x1[i],
                "x2": x2[i],
                "bivariate_logpdf": log_bivar[i],
                "log_rate_1": log_rate_1[i],
                "log_rate_2": log_rate_2[i],
                "lambda_pop": lambda_pop,
                "mixed_log_rate": mixed_rate[i],
                "mix_mass": mix_mass[i],
                "mix_redshift": mix_redshift[i],
                "pl_pl_logpdf": pl_pl_logpdf[i],
                "pl_pl_g_logpdf": pl_pl_g_logpdf[i],
                "pl_g_z_logpdf": pl_g_z_logpdf[i],
                "g_g_z_logpdf": g_g_z_logpdf[i],
                "m1_pair": m1_pair[i],
                "m2_pair": m2_pair[i],
                "paired_dip_logpdf": paired_dip_logpdf[i],
                "paired_dip_general_logpdf": paired_dip_general_logpdf[i],
                "paired_farah_logpdf": paired_farah_logpdf[i],
                "paired_bplmulti_logpdf": paired_bplmulti_logpdf[i],
                "paired_triple_logpdf": paired_triple_logpdf[i],
                "paired_bplmulti_conditioned_logpdf": paired_bplmulti_conditioned_logpdf[i],
                "bin_model_logpdf": bin_model_logpdf[i],
                "chi1": chi1[i],
                "chi2": chi2[i],
                "cos1": cos1[i],
                "cos2": cos2[i],
                "mass1_source": mass1_source[i],
                "mass2_source": mass2_source[i],
                "spin_gaussian_logpdf": spin_gaussian_logpdf[i],
                "spin_evolving_logpdf": spin_evolving_logpdf[i],
                "spin_beta_gaussian_logpdf": spin_beta_gauss_logpdf[i],
                "spin_beta_beta_logpdf": spin_beta_beta_logpdf[i],
                "domega220": domega220[i],
                "dtau220": dtau220[i],
                "pseob_logpdf": pseob_logpdf[i],
                "eco_logpdf": eco_logpdf[i],
            }
        )

    path = outdir / "generated_reference_priors_rates.csv"
    _write_rows(
        path,
        (
            "i",
            "x",
            "highpass",
            "lowpass",
            "notch",
            "mixed_linear",
            "mixed_sigmoid",
            "lowpass_evolving_logpdf",
            "M_abs",
            "abs_l_powerlaw_logpdf",
            "abs_l_powerlaw_cdf",
            "x1",
            "x2",
            "bivariate_logpdf",
            "log_rate_1",
            "log_rate_2",
            "lambda_pop",
            "mixed_log_rate",
            "mix_mass",
            "mix_redshift",
            "pl_pl_logpdf",
            "pl_pl_g_logpdf",
            "pl_g_z_logpdf",
            "g_g_z_logpdf",
            "m1_pair",
            "m2_pair",
            "paired_dip_logpdf",
            "paired_dip_general_logpdf",
            "paired_farah_logpdf",
            "paired_bplmulti_logpdf",
            "paired_triple_logpdf",
            "paired_bplmulti_conditioned_logpdf",
            "bin_model_logpdf",
            "chi1",
            "chi2",
            "cos1",
            "cos2",
            "mass1_source",
            "mass2_source",
            "spin_gaussian_logpdf",
            "spin_evolving_logpdf",
            "spin_beta_gaussian_logpdf",
            "spin_beta_beta_logpdf",
            "domega220",
            "dtau220",
            "pseob_logpdf",
            "eco_logpdf",
        ),
        rows,
    )
    return [path]


def generate_catalog_smoke(outdir: pathlib.Path) -> list[pathlib.Path]:
    """Generate dependency-light catalog luminosity-function fixtures."""

    cosmology = _load_lightweight_cosmology_module()

    class _CosmologyProxy:
        little_h = 0.677

    sch = cosmology.galaxy_MF(band="K-glade+")
    sch.build_MF(_CosmologyProxy())
    epsilon = 0.8
    abs_rate = cosmology.log_powerlaw_absM_rate(epsilon=epsilon)
    sch.build_effective_number_density_interpolant(epsilon)

    M_abs = np.array([-25.0, -22.0, -18.0])
    z = np.array([0.0, 0.5, 1.0])
    Mthr = np.array([-26.0, -22.0, -18.0])
    phistar, Mstar = sch.get_evol_phi_Mstar(z)
    log_lf = sch.log_evaluate(M_abs.copy(), z)
    lf = sch.evaluate(M_abs.copy(), z)
    log_abs_rate = abs_rate.log_evaluate(sch, M_abs.copy())
    abs_rate_eval = abs_rate.evaluate(sch, M_abs.copy())
    background_density = sch.background_effective_galaxy_density(Mthr, z)

    rows = []
    for i in range(len(M_abs)):
        rows.append(
            {
                "i": i + 1,
                "band": "K-glade+",
                "little_h": _CosmologyProxy.little_h,
                "Mminobs": sch.Mminobs,
                "Mmaxobs": sch.Mmaxobs,
                "Mstarobs": sch.Mstarobs,
                "phistarobs": sch.phistarobs,
                "epsilon": epsilon,
                "M_abs": M_abs[i],
                "redshift": z[i],
                "Mthr": Mthr[i],
                "phistar_z": phistar[i],
                "Mstar_z": Mstar[i],
                "log_luminosity_function": log_lf[i],
                "luminosity_function": lf[i],
                "log_abs_magnitude_rate": log_abs_rate[i],
                "abs_magnitude_rate": abs_rate_eval[i],
                "background_effective_density": background_density[i],
            }
        )

    path = outdir / "generated_reference_catalog_luminosity.csv"
    _write_rows(
        path,
        (
            "i",
            "band",
            "little_h",
            "Mminobs",
            "Mmaxobs",
            "Mstarobs",
            "phistarobs",
            "epsilon",
            "M_abs",
            "redshift",
            "Mthr",
            "phistar_z",
            "Mstar_z",
            "log_luminosity_function",
            "luminosity_function",
            "log_abs_magnitude_rate",
            "abs_magnitude_rate",
            "background_effective_density",
        ),
        rows,
    )
    return [path]


def not_yet_available(name: str) -> None:
    raise SystemExit(
        f"Fixture suite '{name}' is not implemented yet. Add it when the corresponding "
        "Julia migration phase starts, and keep the generated files small and documented."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("core", "stochastic-smoke", "simulation-utils-smoke", "priors-rates-smoke", "catalog-smoke", "all-small"),
        default="core",
        help="Fixture suite to generate.",
    )
    parser.add_argument("--outdir", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not PYTHON_REF.exists():
        raise SystemExit(f"Python reference checkout not found at {PYTHON_REF}")

    generated: list[pathlib.Path] = []
    if args.suite in ("core", "all-small"):
        generated.extend(generate_core(args.outdir))
    if args.suite in ("stochastic-smoke", "all-small"):
        generated.extend(generate_stochastic_smoke(args.outdir))
    if args.suite in ("simulation-utils-smoke", "all-small"):
        generated.extend(generate_simulation_utils_smoke(args.outdir))
    if args.suite in ("priors-rates-smoke", "all-small"):
        generated.extend(generate_priors_rates_smoke(args.outdir))
    if args.suite == "catalog-smoke":
        generated.extend(generate_catalog_smoke(args.outdir))

    for path in generated:
        try:
            print(path.relative_to(REPO_ROOT))
        except ValueError:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

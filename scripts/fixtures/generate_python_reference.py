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
    sys.modules.setdefault("icarogw.cosmology", cosmology_stub)
    package.cosmology = sys.modules["icarogw.cosmology"]

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

    m1 = 36.0
    m2 = 21.0
    luminosity = 3.0128e28
    magnitude = -19.2
    chi1 = 0.4
    chi2 = 0.7
    cos1 = 0.3
    cos2 = -0.2
    q = m2 / m1

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
    import scipy.special  # noqa: F401 - ensures priors.py can access scipy.special

    priors = _load_reference_module("icarogw.reference_priors", "icarogw/priors.py")
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
        not_yet_available(args.suite)

    for path in generated:
        try:
            print(path.relative_to(REPO_ROOT))
        except ValueError:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

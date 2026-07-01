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
import pathlib
import sys
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


def _write_rows(path: pathlib.Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def generate_core(outdir: pathlib.Path) -> list[pathlib.Path]:
    """Regenerate lightweight conversion fixtures.

    The existing repository already contains broader core fixtures. This suite
    is intentionally tiny and dependency-light so contributors can verify their
    Python reference environment before adding heavier catalog/stochastic
    fixture suites.
    """

    _prepend_python_reference()
    conversions = importlib.import_module("icarogw.conversions")

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
            "chirp_mass": conversions.chirp_mass(m1, m2),
            "mass_ratio": conversions.mass_ratio(m1, m2),
            "f_gw_isco": conversions.f_GW_ISCO(m1, m2),
            "L": luminosity,
            "L2M": conversions.L2M(luminosity),
            "M": magnitude,
            "M2L": conversions.M2L(magnitude),
            "chi1": chi1,
            "chi2": chi2,
            "cos1": cos1,
            "cos2": cos2,
            "q": q,
            "chi_eff": conversions.chi_eff_from_spins(chi1, chi2, cos1, cos2, q),
            "chi_p": conversions.chi_p_from_spins(chi1, chi2, cos1, cos2, q),
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


def not_yet_available(name: str) -> None:
    raise SystemExit(
        f"Fixture suite '{name}' is not implemented yet. Add it when the corresponding "
        "Julia migration phase starts, and keep the generated files small and documented."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("core", "stochastic-smoke", "catalog-smoke", "all-small"),
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

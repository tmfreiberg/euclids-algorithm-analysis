"""Tests for :mod:`euclid_analysis.constants`.

The expected values are those quoted in the exposition; tolerances are loose
enough to be robust to the last displayed digit but tight enough to catch a
mis-transcribed formula.
"""

from __future__ import annotations

import pytest

from euclid_analysis import constants as C


@pytest.mark.parametrize(
    ("value", "expected", "tol"),
    [
        (C.LAMBDA_DIXON, 0.8427659, 1e-6),          # lambda = 2 log2 / zeta(2)
        (C.LAMBDA_DIXON ** 2, 0.7102543, 1e-6),     # lambda^2 (quoted in text)
        (C.PORTER_CONSTANT, 1.4670780, 1e-6),       # Porter's constant C_P
        (C.NU_NORTON - 0.5, -0.4346485, 1e-6),      # nu - 1/2
        (C.NU_NORTON_COPRIME - 0.5, 0.0456951, 1e-6),  # nu_1 - 1/2
        (C.NU_NORTON_COPRIME - C.NU_NORTON, 0.4803436, 1e-6),  # nu_1 - nu
        (C.ETA_HENSLEY, 0.5160524, 1e-7),           # Hensley's constant
        (C.DELTA_KAPPA, 0.334, 1e-3),               # delta_kappa (rounded)
        (C.KAPPA_VAR, -0.1, 1e-9),
        (C.KAPPA_VAR_COPRIME, -0.434, 1e-3),
    ],
)
def test_constant_values(value, expected, tol):
    assert value == pytest.approx(expected, abs=tol)


def test_delta_kappa_is_rounded_to_three_places():
    assert round(C.DELTA_KAPPA, 3) == C.DELTA_KAPPA

r"""Constants governing the mean and variance of Euclid's step count.

For ``(a, b)`` drawn uniformly from ``1 <= b < a <= N``, the number of division
steps ``T(a, b)`` is asymptotically normal as ``N -> infinity`` with

* mean close to ``lambda * log(N) + nu - 1/2``, and
* variance close to ``eta * log(N) + kappa``.

The constants associated with the *mean* have closed forms and are computed
here to full floating-point accuracy.  The constants associated with the
*variance* do not: ``eta`` is only known numerically (Lhote), and the
subdominant constant ``kappa`` is the object this project estimates empirically.

Symbol summary
--------------
``LAMBDA_DIXON`` (:math:`\lambda`)
    ``2 log 2 / zeta(2)``; the reciprocal of Lévy's constant, governing the
    leading term of the mean.  Named here in honour of Dixon.
``PORTER_CONSTANT`` (:math:`C_P`)
    Porter's constant, the subdominant constant in the one-dimensional mean.
``NU_NORTON`` (:math:`\nu`), ``NU_NORTON_COPRIME``
    Norton's subdominant constants for the two-dimensional mean (all pairs and
    coprime pairs respectively).
``ETA_HENSLEY`` (:math:`\eta`)
    Hensley's leading constant in the variance, to seven digits (Lhote).
``KAPPA_VAR`` (:math:`\kappa`), ``KAPPA_VAR_COPRIME``
    Empirical estimates of the subdominant constant in the variance.

The numeric literals for :data:`EULER_MASCHERONI`, :data:`ZETA_PRIME_2`, and
:data:`ZETA_DOUBLE_PRIME_2` are retained exactly as in the original project so
that every downstream constant reproduces the published values bit for bit.
"""

from __future__ import annotations

import numpy as np
from scipy.special import zeta

__all__ = [
    "EULER_MASCHERONI",
    "ZETA_PRIME_2",
    "ZETA_DOUBLE_PRIME_2",
    "LAMBDA_DIXON",
    "PORTER_CONSTANT",
    "NU_NORTON",
    "NU_NORTON_COPRIME",
    "ETA_HENSLEY",
    "KAPPA_VAR",
    "DELTA_KAPPA",
    "KAPPA_VAR_COPRIME",
]

# --- Auxiliary constants needed to define the others -----------------------

#: Euler--Mascheroni constant, ``gamma``.
EULER_MASCHERONI: float = 0.57721566490153286060

#: ``zeta'(2)``, the first derivative of the Riemann zeta function at 2.
ZETA_PRIME_2: float = -0.93754825431584375370

#: ``zeta''(2)``, the second derivative of the Riemann zeta function at 2.
ZETA_DOUBLE_PRIME_2: float = 1.98928

# --- Constants for the mean (one- and two-dimensional analyses) -------------

#: ``lambda = 2 log 2 / zeta(2)``, the reciprocal of Lévy's constant.
LAMBDA_DIXON: float = 2 * np.log(2) / zeta(2)

#: Porter's constant ``C_P``, the subdominant constant in the 1-D mean.
PORTER_CONSTANT: float = (
    (np.log(2) / zeta(2))
    * (3 * np.log(2) + 4 * EULER_MASCHERONI - 4 * ZETA_PRIME_2 / zeta(2) - 2)
    - 0.5
)

# --- Subdominant constants for the two-dimensional mean (Norton) ------------

#: Norton's subdominant constant for the mean over *all* pairs.
NU_NORTON: float = -1 + LAMBDA_DIXON * (
    2 * EULER_MASCHERONI + 1.5 * np.log(2) - 1.5 - ZETA_PRIME_2 / zeta(2)
)

#: Norton's subdominant constant for the mean over *coprime* pairs.
NU_NORTON_COPRIME: float = NU_NORTON - LAMBDA_DIXON * ZETA_PRIME_2 / zeta(2)

# --- Constants for the two-dimensional variance -----------------------------

#: Hensley's leading constant ``eta`` in the variance, to seven digits (Lhote).
ETA_HENSLEY: float = 0.5160524

#: Empirical estimate of the subdominant constant ``kappa`` in the variance
#: (all pairs).  This project's numerical investigation supports a value near
#: ``-0.1``.
KAPPA_VAR: float = -0.1

#: The offset relating the all-pairs and coprime-pairs subdominant constants,
#: rounded to three decimals as in the original analysis.
DELTA_KAPPA: float = np.around(
    ETA_HENSLEY * ZETA_PRIME_2 / zeta(2)
    + (LAMBDA_DIXON ** 2)
    * ((ZETA_DOUBLE_PRIME_2 / zeta(2)) - (ZETA_PRIME_2 / zeta(2)) ** 2),
    3,
)

#: Empirical estimate of the subdominant constant in the variance (coprime
#: pairs), derived from :data:`KAPPA_VAR` and :data:`DELTA_KAPPA`.
KAPPA_VAR_COPRIME: float = np.around(KAPPA_VAR - DELTA_KAPPA, 3)

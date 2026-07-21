"""Fibonacci-number implementations used in the worst-case analysis.

Consecutive Fibonacci numbers are the worst case for Euclid's algorithm: if
``T(a, b) = n`` then ``a >= f_{n+2}`` and ``b >= f_{n+1}``, with equality for
consecutive Fibonacci numbers.  This module gathers several ways of computing
``f_n``, chosen to illustrate a spectrum of algorithmic ideas rather than to
provide a single "best" routine:

``naive_fib``
    The direct exponential-time recursion, useful only as a cautionary tale.
``fibonacci_list``
    Bottom-up tabulation returning ``f_0, ..., f_n`` (dynamic programming).
``fib``
    Bottom-up memoisation using two rolling variables: ``O(n)`` time, ``O(1)``
    space -- the archetypal dynamic-programming solution.
``binet_fib``
    The closed form via the golden ratio (Binet's formula), exact only up to
    floating-point precision.
``time_fibonacci_list``
    Instrumentation used to produce the timing plot in the exposition.

The sequence is indexed so that ``f_0 = 0`` and ``f_1 = 1``.
"""

from __future__ import annotations

from timeit import default_timer as timer

__all__ = [
    "naive_fib",
    "fibonacci_list",
    "fib",
    "binet_fib",
    "time_fibonacci_list",
    "PHI",
]

#: The golden ratio, ``(1 + sqrt(5)) / 2``.
PHI: float = (1 + 5 ** 0.5) / 2


def naive_fib(n: int) -> int:
    """Return ``f_n`` by direct recursion (exponential time).

    Computing ``f_n`` this way expands a binary tree with roughly ``f_n``
    leaves, so the running time is of order ``phi ** n``.  It is included to
    contrast with the linear-time alternatives; do not use it for ``n`` beyond
    the low twenties.

    Parameters
    ----------
    n : int
        Index.  Negative indices are reflected via ``naive_fib(-n)``.

    Returns
    -------
    int
        The ``n``-th Fibonacci number.

    Examples
    --------
    >>> naive_fib(10)
    55
    """
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n > 1:
        return naive_fib(n - 1) + naive_fib(n - 2)
    return naive_fib(-n)


def fibonacci_list(n: int) -> list[int]:
    """Return the list ``[f_0, f_1, ..., f_n]`` by bottom-up tabulation.

    This is the "tabulation" flavour of dynamic programming: every value is
    computed once, in order, and retained.  It is the natural choice when the
    whole prefix of the sequence is wanted.

    Parameters
    ----------
    n : int
        Non-negative largest index to compute.

    Returns
    -------
    list of int
        ``f_0`` through ``f_n`` inclusive (length ``n + 1``).

    Raises
    ------
    ValueError
        If ``n`` is negative.

    Examples
    --------
    >>> fibonacci_list(10)
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    """
    if n < 0:
        raise ValueError("fibonacci_list requires n >= 0")
    values: list[int] = []
    for k in range(n + 1):
        if k == 0 or k == 1:
            values.append(k)
        else:
            values.append(values[k - 1] + values[k - 2])
    return values


def fib(n: int) -> int:
    """Return ``f_n`` in ``O(n)`` time and ``O(1)`` space (memoisation).

    Two rolling variables ``a`` and ``b`` hold ``f_{k-2}`` and ``f_{k-1}`` and
    are updated in place -- the archetypal memoised dynamic-programming
    computation, requiring only about ``n`` additions and no growing list.

    Parameters
    ----------
    n : int
        Index.  Negative indices are reflected via ``fib(-n)``.

    Returns
    -------
    int
        The ``n``-th Fibonacci number.

    Examples
    --------
    >>> [fib(k) for k in range(11)]
    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    """
    a, b = 0, 1  # placeholders for f_{k-2} and f_{k-1}
    if n == 0:
        return a
    if n == 1:
        return b
    if n < 0:
        return fib(-n)
    c = b
    for _ in range(n - 1):
        c = a + b
        a = b
        b = c
    return c


def binet_fib(n: int) -> int:
    """Return ``f_n`` via Binet's closed form, rounded to the nearest integer.

    Binet's formula gives ``f_n = (phi**n - (-phi)**(-n)) / sqrt(5)``.  Because
    ``|(-phi)**(-n)| < 1/2`` for ``n >= 0``, rounding ``phi**n / sqrt(5)`` to
    the nearest integer recovers ``f_n`` exactly -- subject to floating-point
    precision, which degrades for large ``n`` (roughly ``n > 70`` on IEEE-754
    doubles).

    Parameters
    ----------
    n : int
        Non-negative index.

    Returns
    -------
    int
        The ``n``-th Fibonacci number (exact within floating-point range).

    Examples
    --------
    >>> binet_fib(10)
    55
    """
    return round(PHI ** n / 5 ** 0.5)


def time_fibonacci_list(n: int = 100) -> tuple[list[int], list[float]]:
    """Compute ``f_0, ..., f_n`` by tabulation, timing each individual step.

    This reproduces the measurement behind the timing plot in the exposition.
    It is separated into a function (rather than run at import time, as in the
    original single-file script) so that importing the package has no side
    effects and the measurement can be repeated on demand.

    Parameters
    ----------
    n : int, optional
        Largest index to compute (default ``100``).

    Returns
    -------
    tuple of (list of int, list of float)
        The Fibonacci prefix ``[f_0, ..., f_n]`` and a parallel list of
        per-step wall-clock durations in seconds.
    """
    values: list[int] = []
    durations: list[float] = []
    for k in range(n + 1):
        start = timer()
        if k == 0 or k == 1:
            values.append(k)
        else:
            values.append(values[k - 1] + values[k - 2])
        durations.append(timer() - start)
    return values, durations

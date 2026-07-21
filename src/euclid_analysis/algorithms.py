"""Implementations of Euclid's algorithm and closely related routines.

This module collects several variants of Euclid's algorithm, each exposing a
different amount of the algorithm's internal state:

``gcd``
    The greatest common divisor only.
``gcdn``
    The greatest common divisor of an arbitrary number of integers.
``count_steps`` / ``gcd_steps``
    The greatest common divisor together with the number of division steps.
``remainders``
    The full remainder sequence.
``quotients_remainders``
    The full quotient *and* remainder sequences.
``write_euclid``
    A human-readable transcript of the whole computation.

The number of division steps is the central object of study in this project.
For integers ``a`` and ``b`` it is written ``T(a, b)`` throughout the
accompanying exposition and equals ``len(remainders(a, b)) - 1``.

All functions accept negative integers.  Because ``gcd(a, b) == gcd(|a|, |b|)``
and the *number* of division steps is likewise invariant under sign changes,
the sign of the inputs never affects the step count -- only the transcript in
:func:`write_euclid` displays signs.

Notes
-----
The recursive formulations mirror the mathematical recurrence
``gcd(a, b) = gcd(b, a mod b)`` as directly as possible; they are intended to
read like the definitions in the exposition rather than to be maximally fast.
The recursion depth is ``T(a, b)``, which grows only logarithmically in the
inputs (the worst case is consecutive Fibonacci numbers), so Python's default
recursion limit is never a practical concern here.
"""

from __future__ import annotations

__all__ = [
    "gcd",
    "gcdn",
    "count_steps",
    "gcd_steps",
    "remainders",
    "quotients_remainders",
    "write_euclid",
]


def gcd(a: int, b: int) -> int:
    """Return the greatest common divisor of ``a`` and ``b``.

    Implements Euclid's algorithm via the recurrence
    ``gcd(a, b) = gcd(b, a mod b)`` with base case ``gcd(a, 0) = |a|``.

    Parameters
    ----------
    a, b : int
        Any integers (may be negative or zero).

    Returns
    -------
    int
        The non-negative greatest common divisor.  By convention
        ``gcd(0, 0) == 0``.

    Examples
    --------
    >>> gcd(1011, 69)
    3
    >>> gcd(-1011, -69)
    3
    >>> gcd(0, 0)
    0
    """
    if b == 0:
        return abs(a)
    return gcd(b, a % b)


def gcdn(*ntuple: int) -> int:
    """Return the greatest common divisor of arbitrarily many integers.

    The greatest common divisor of an ``n``-tuple is defined, for ``n >= 2``,
    by the property that an integer ``d`` divides every ``a_i`` if and only if
    it divides ``gcdn(a_1, ..., a_n)``.  This yields the recurrence
    ``gcdn(a_1, ..., a_n) = gcd(a_1, gcdn(a_2, ..., a_n))``.

    Parameters
    ----------
    *ntuple : int
        Two or more integers.

    Returns
    -------
    int
        The non-negative greatest common divisor of all arguments.

    Raises
    ------
    TypeError
        If fewer than two arguments are supplied.

    Examples
    --------
    >>> gcdn(12, 18, 30)
    6
    >>> gcdn(7, 14)
    7
    """
    if len(ntuple) < 2:
        raise TypeError("gcdn() requires at least two arguments")
    if len(ntuple) == 2:
        return gcd(ntuple[0], ntuple[1])
    return gcd(ntuple[0], gcdn(*ntuple[1:]))


def count_steps(a: int, b: int) -> tuple[int, int]:
    """Return ``(gcd(a, b), T(a, b))``: the gcd and the number of steps.

    ``T(a, b)`` is the number of division steps
    ``(u, v) -> (v, u mod v)`` performed by Euclid's algorithm.

    Parameters
    ----------
    a, b : int
        Any integers.

    Returns
    -------
    tuple of (int, int)
        A pair ``(g, s)`` where ``g == gcd(a, b)`` and ``s == T(a, b)``.

    Examples
    --------
    >>> count_steps(1011, 69)
    (3, 5)
    >>> count_steps(5, 0)
    (5, 0)
    """
    return gcd_steps(a, b, 0)


def gcd_steps(a: int, b: int, i: int = 0) -> tuple[int, int]:
    """Recursively compute the gcd while accumulating the step count in ``i``.

    This is the low-level recursive worker underlying :func:`count_steps`.  It
    is retained with its original three-argument signature (now with a default
    of ``i=0``) because the exposition refers to it directly.  Prefer
    :func:`count_steps` in new code.

    Parameters
    ----------
    a, b : int
        Any integers.
    i : int, optional
        The running step count; callers should leave this at its default of
        ``0``.  It is incremented once per recursive division.

    Returns
    -------
    tuple of (int, int)
        The pair ``(gcd(a, b), T(a, b) + i)``.  With the default ``i=0`` the
        second entry is exactly ``T(a, b)``.
    """
    if b == 0:
        return abs(a), i
    i += 1
    return gcd_steps(b, a % b, i)


def remainders(a: int, b: int, r: list[int] | None = None) -> list[int]:
    """Return the remainder sequence ``[r_0, r_1, ..., r_n]`` of Euclid's algorithm.

    With ``a = r_0`` and ``b = r_1``, the sequence satisfies
    ``r_1 > r_2 > ... > r_n > r_{n+1} = 0`` (for non-negative inputs), and the
    greatest common divisor is ``abs(r[-1])``.  The number of steps is
    ``T(a, b) == len(remainders(a, b)) - 1``.

    Parameters
    ----------
    a, b : int
        Any integers.
    r : list of int, optional
        Accumulator for the sequence.  Callers should leave this as ``None``
        (a fresh list is created on each top-level call); it exists to support
        the tail recursion.  Passing a non-empty list prepends its contents.

    Returns
    -------
    list of int
        The remainder sequence, beginning with ``a`` and ``b``.

    Examples
    --------
    >>> remainders(1011, 69)
    [1011, 69, 45, 24, 21, 3]
    """
    if r is None:
        r = []
    r.append(a)
    if b == 0:
        return r
    return remainders(b, a % b, r)


def quotients_remainders(
    a: int,
    b: int,
    q: list[int] | None = None,
    r: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Return the quotient and remainder sequences of Euclid's algorithm.

    For ``i = 1, ..., n`` the algorithm produces
    ``r_{i-1} = q_i * r_i + r_{i+1}``.  This returns the quotients
    ``[q_1, ..., q_n]`` and the remainder sequence ``[r_0, r_1, ..., r_n]``.

    Parameters
    ----------
    a, b : int
        Any integers.
    q, r : list of int, optional
        Accumulators for the quotient and remainder sequences.  Callers should
        leave both as ``None``; they exist to support the tail recursion.

    Returns
    -------
    tuple of (list of int, list of int)
        The pair ``(quotients, remainders)``.  The remainder list is identical
        to :func:`remainders`.

    Examples
    --------
    >>> quotients_remainders(1011, 69)
    ([14, 1, 1, 1, 7], [1011, 69, 45, 24, 21, 3])
    """
    if q is None:
        q = []
    if r is None:
        r = []
    r.append(a)
    if b == 0:
        return q, r
    quotient, remainder = divmod(a, b)
    q.append(quotient)
    return quotients_remainders(b, remainder, q, r)


def write_euclid(a: int, b: int) -> str:
    """Return an aligned, human-readable transcript of Euclid's algorithm.

    Each line has the form ``r_{i-1} = q_i * r_i +/- r_{i+1}`` together with the
    corresponding gcd identity ``gcd(r_{i-1}, r_i) = gcd(r_i, r_{i+1})``.  The
    transcript ends with a summary line giving ``gcd(a, b)`` and ``T(a, b)``.

    Unlike the original implementation, which printed directly to standard
    output, this version *returns* the transcript as a string so that it can be
    tested, embedded in documents, or printed by the caller.

    Parameters
    ----------
    a, b : int
        Any integers.

    Returns
    -------
    str
        The formatted transcript.  Negative operands are parenthesised and a
        negative remainder ``+ -r`` is rendered as ``- r`` for readability.

    Examples
    --------
    >>> print(write_euclid(1011, 69))
    1011 = 14*69 + 45 	 ∴ gcd(1011,69) = gcd(69,45)
      69 =  1*45 + 24 	 ∴   gcd(69,45) = gcd(45,24)
      45 =  1*24 + 21 	 ∴   gcd(45,24) = gcd(24,21)
      24 =  1*21 +  3 	 ∴   gcd(24,21) = gcd(21,3)
      21 =  7* 3 +  0 	 ∴    gcd(21,3) = gcd(3,0)
    <BLANKLINE>
    gcd(1011,69) = 3, T(1011,69) = 5
    <BLANKLINE>
    """
    q, r = quotients_remainders(a, b, [], [])

    # Build the right-hand remainder column, terminated by the final 0.
    tail = r[2:]
    tail.append(0)

    left, middle, quots = r, r[1:], q

    # Column of gcd identities, one per division line plus the closing pair.
    gcd_col = [f"gcd({a},{b})", f"gcd({b},{tail[0]})"]
    for i in range(len(tail) - 1):
        gcd_col.append(f"gcd({tail[i]},{tail[i + 1]})")

    # Parenthesise negative operands for display.
    middle_disp: list[str | int] = [f"({v})" if v < 0 else v for v in middle]
    quots_disp: list[str | int] = [f"({v})" if v < 0 else v for v in quots]

    # Render "a = bq - r" rather than "a = bq + -r" when the remainder is < 0.
    signs: list[str] = []
    display_tail: list[int] = []
    for value in tail:
        if value < 0:
            display_tail.append(-value)
            signs.append("-")
        else:
            display_tail.append(value)
            signs.append("+")

    # Column widths for alignment.
    width_left = max(len(str(v)) for v in left)
    width_middle = max(len(str(v)) for v in middle_disp)
    width_quot = max(len(str(v)) for v in quots_disp)
    width_rem = max(len(str(v)) for v in display_tail)
    width_gcd_left = max(len(s) for s in gcd_col[:-2])
    width_gcd_right = max(len(s) for s in gcd_col[1:])

    lines: list[str] = []
    if len(left) > 1:
        for i in range(len(display_tail)):
            lines.append(
                f"{left[i]:>{width_left}} = "
                f"{quots_disp[i]:>{width_quot}}*{middle_disp[i]:>{width_middle}} "
                f"{signs[i]} {display_tail[i]:>{width_rem}} \t ∴ "
                f"{gcd_col[i]:>{width_gcd_left}} = "
                f"{gcd_col[i + 1]:<{width_gcd_right}}"
            )
    lines.append(
        f"\ngcd({a},{b}) = {abs(r[-1])}, T({a},{b}) = {len(r) - 1}\n"
    )
    return "\n".join(lines)
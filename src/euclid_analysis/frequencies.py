"""Frequency tables of Euclid's step count over ranges of integer pairs.

Two regimes are studied:

**One-dimensional** (:func:`heilbronn`).
    For each fixed ``a``, count how many ``b`` in ``[1, a]`` give each step
    count ``T(a, b) = s``.  This exposes the distribution of the step count as
    ``b`` ranges over the residues (or the totatives, when restricted to
    ``gcd(a, b) = 1``) of a single ``a``.

**Two-dimensional** (:func:`euclid_alg_frequencies`).
    Count over the triangular region ``0 < b < a < N`` simultaneously, for a
    sequence of checkpoints ``N``.  This yields, for each ``N``, both the
    distribution of gcd values and the distribution of step counts, and is
    built so that an existing computation can be *extended* to larger ``N``
    without recomputing the part already done.

Data model
----------
A **frequency dictionary** maps an integer key to a count, e.g. ``{s: count}``
mapping a step count to the number of pairs achieving it, or ``{g: count}``
mapping a gcd value to its multiplicity.

A **meta dictionary** maps an outer key (an ``a`` in the 1-D case, or a
checkpoint ``N`` in the 2-D case) to a frequency dictionary.  These meta
dictionaries are what :mod:`euclid_analysis.dataio` serialises to CSV and what
:mod:`euclid_analysis.statistics` summarises.
"""

from __future__ import annotations

from .algorithms import gcd_steps

__all__ = [
    "dictionary_sort",
    "heilbronn",
    "euclid_alg_frequencies",
]

# Type aliases for readability.
FrequencyDict = dict[int, int]
MetaDict = dict[int, FrequencyDict]


def dictionary_sort(dictionary: dict) -> dict:
    """Return a shallow copy of ``dictionary`` with keys in sorted order.

    Insertion order is meaningful for :class:`dict` in modern Python, and
    several downstream routines (statistics, plotting) iterate keys in their
    natural order.  This helper guarantees that order without mutating the
    input.

    Parameters
    ----------
    dictionary : dict
        A dictionary whose keys are mutually comparable.

    Returns
    -------
    dict
        A new dictionary with the same items, iterated in ascending key order.

    Examples
    --------
    >>> dictionary_sort({3: 'c', 1: 'a', 2: 'b'})
    {1: 'a', 2: 'b', 3: 'c'}
    """
    return {key: dictionary[key] for key in sorted(dictionary.keys())}


def heilbronn(
    gcdlist: list[int],
    list1: list[int],
) -> tuple[MetaDict, MetaDict]:
    """Count step-count frequencies for each ``a``, over ``b`` in ``[1, a]``.

    For every ``a`` in ``list1`` and every ``b`` in ``[1, a]`` the pair's step
    count ``s = T(a, b)`` and gcd ``g = gcd(a, b)`` are computed.  Two meta
    dictionaries are returned, both keyed by ``a`` with frequency-dictionary
    values ``{s: count}``:

    * the **restricted** table counts only ``b`` with ``g in gcdlist``
      (typically ``gcdlist == [1]``, i.e. ``b`` a totative of ``a``);
    * the **all** table counts every ``b`` in ``[1, a]``.

    Parameters
    ----------
    gcdlist : list of int
        The gcd values to admit in the restricted table.  The common choice
        ``[1]`` restricts to coprime pairs.
    list1 : list of int
        The values of ``a`` to tabulate.  It is not mutated; a sorted copy is
        used internally.

    Returns
    -------
    tuple of (MetaDict, MetaDict)
        ``(restricted, all_)`` -- the restricted and unrestricted meta
        dictionaries described above.

    Examples
    --------
    >>> restricted, all_ = heilbronn([1], [5])
    >>> all_[5]
    {1: 2, 2: 2, 3: 1}
    >>> restricted[5]
    {1: 1, 2: 2, 3: 1}
    """
    output_dictionary_restricted: MetaDict = {}
    output_dictionary_all: MetaDict = {}
    for a in sorted(list1):
        output_dictionary_restricted[a] = {}
        output_dictionary_all[a] = {}
        for b in range(1, a + 1):
            g, s = gcd_steps(a, b, 0)
            if g in gcdlist:
                if s in output_dictionary_restricted[a]:
                    output_dictionary_restricted[a][s] += 1
                else:
                    output_dictionary_restricted[a][s] = 1
            if s in output_dictionary_all[a]:
                output_dictionary_all[a][s] += 1
            else:
                output_dictionary_all[a][s] = 1
    return output_dictionary_restricted, output_dictionary_all


def euclid_alg_frequencies(
    gcdlist: list[int],
    list1: list[int],
    list2: list[int],
    gcd_dictionary: MetaDict,
    steps_dictionary: MetaDict,
    steps_dictionary_all: MetaDict,
) -> tuple[MetaDict, MetaDict, MetaDict]:
    """Accumulate gcd- and step-count frequencies over ``0 < b < a < N``.

    For a strictly increasing list of checkpoints ``list2`` the region
    ``0 < b < a < N`` is swept for each ``N`` in ``list2[1:]``, and three meta
    dictionaries keyed by ``N`` are produced.  With ``A``, ``B``, ``C`` denoting
    the returned dictionaries and ``N`` a key:

    * ``A[N] = {g: #{0 < b < a < N : gcd(a, b) = g}}`` (gcd frequencies);
    * ``B[N] = {s: #{0 < b < a < N, gcd(a, b) in gcdlist : T(a, b) = s}}``
      (step-count frequencies restricted by ``gcdlist``);
    * ``C[N] = {s: #{0 < b < a < N : T(a, b) = s}}`` (all step-count
      frequencies).

    **Incremental extension.**  Building these tables for large ``N`` is
    expensive, so the computation can be resumed.  To extend existing tables
    ``A``, ``B``, ``C``, pass them as the three dictionary arguments and pass
    their checkpoints as ``list1`` (all that matters is ``list1[-1] ==
    list2[0]``); ``list2[1:]`` are the new checkpoints to append.  For a fresh
    computation, pass ``list1 == []`` and empty dictionaries, with ``list2``
    beginning at the lowest checkpoint.

    Parameters
    ----------
    gcdlist : list of int
        gcd values admitted in the restricted step-count table ``B``.  Use
        ``[1]`` for coprime pairs; an empty list admits every gcd.
    list1 : list of int
        Checkpoints already computed (empty for a fresh run).  Not mutated.
    list2 : list of int
        Checkpoints to compute up to, in the region ``0 < b < a < N``.  Not
        mutated; a sorted copy is used.
    gcd_dictionary, steps_dictionary, steps_dictionary_all : MetaDict
        Existing tables ``A``, ``B``, ``C`` to extend (empty for a fresh run).
        Not mutated in place; sorted copies seed the result.

    Returns
    -------
    tuple of (MetaDict, MetaDict, MetaDict)
        The updated ``(A, B, C)`` meta dictionaries.

    Notes
    -----
    The two inner double loops cover, respectively, the newly added triangular
    block ``[list2[k], list2[k+1])^2`` (with ``b < a``) and the rectangular
    strip pairing earlier ``b`` with the new ``a`` range, so that together they
    account for exactly the pairs entering the region as ``N`` grows from
    ``list2[k]`` to ``list2[k+1]``.
    """
    checkpoints2 = sorted(list2)
    gcd_frequencies: FrequencyDict = {}
    steps_frequencies: FrequencyDict = {}
    steps_frequencies_all: FrequencyDict = {}
    updated_gcd_dictionary = dictionary_sort(gcd_dictionary)
    updated_steps_dictionary = dictionary_sort(steps_dictionary)
    updated_steps_dictionary_all = dictionary_sort(steps_dictionary_all)

    if not list1:
        b_0 = checkpoints2[0]
    else:
        checkpoints1 = sorted(list1)
        b_0 = checkpoints1[0]
        # Seed the running counts from the last checkpoint already computed.
        for key, value in updated_gcd_dictionary[checkpoints1[-1]].items():
            gcd_frequencies[key] = value
        for key, value in updated_steps_dictionary[checkpoints1[-1]].items():
            steps_frequencies[key] = value
        for key, value in updated_steps_dictionary_all[checkpoints1[-1]].items():
            steps_frequencies_all[key] = value

    for k in range(len(checkpoints2) - 1):
        lo, hi = checkpoints2[k], checkpoints2[k + 1]
        # New triangular block: b in [lo, hi), a in (b, hi).
        for b in range(lo, hi):
            for a in range(b + 1, hi):
                g, s = gcd_steps(a, b, 0)
                gcd_frequencies[g] = gcd_frequencies.get(g, 0) + 1
                steps_frequencies_all[s] = steps_frequencies_all.get(s, 0) + 1
                if g in gcdlist or not gcdlist:
                    steps_frequencies[s] = steps_frequencies.get(s, 0) + 1
        # Rectangular strip: earlier b in [b_0, lo), new a in [lo, hi).
        for b in range(b_0, lo):
            for a in range(lo, hi):
                g, s = gcd_steps(a, b, 0)
                gcd_frequencies[g] = gcd_frequencies.get(g, 0) + 1
                steps_frequencies_all[s] = steps_frequencies_all.get(s, 0) + 1
                if g in gcdlist or not gcdlist:
                    steps_frequencies[s] = steps_frequencies.get(s, 0) + 1
        updated_gcd_dictionary[hi] = dictionary_sort(gcd_frequencies)
        updated_steps_dictionary[hi] = dictionary_sort(steps_frequencies)
        updated_steps_dictionary_all[hi] = dictionary_sort(steps_frequencies_all)

    return (
        updated_gcd_dictionary,
        updated_steps_dictionary,
        updated_steps_dictionary_all,
    )

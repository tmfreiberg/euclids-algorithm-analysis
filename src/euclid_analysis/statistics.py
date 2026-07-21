"""Summary statistics for frequency dictionaries.

The frequency tables produced by :mod:`euclid_analysis.frequencies` are
distributions of an integer-valued quantity (a step count) given as
``{value: count}``.  This module turns those raw counts into relative
frequencies and moments.

``dictionary_statistics``
    Full summary for a single distribution: relative frequencies, mean, second
    moment, variance, standard deviation, median, and mode(s).
``basic_stats`` / ``dists``
    Apply the summary across a whole meta dictionary, returning either the full
    summaries or just the relative-frequency distributions, keyed as the input.
``tabulate``
    Render a meta dictionary as a :class:`pandas.DataFrame` for display or CSV
    export, with missing cells filled with zeros.

The median is computed directly from the cumulative counts (no expansion of the
data), handling the even- and odd-sized cases separately, and the mode is the
list of all values attaining the maximum frequency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .frequencies import dictionary_sort

__all__ = [
    "dictionary_statistics",
    "basic_stats",
    "dists",
    "tabulate",
]

FrequencyDict = dict[int, int]
MetaDict = dict[int, FrequencyDict]


def dictionary_statistics(dictionary: FrequencyDict) -> dict[str, Any]:
    """Summarise a single frequency distribution.

    The input maps each observed value to the number of times it occurs.  The
    returned summary contains, under fixed string keys:

    ``"dist"``
        Relative frequencies ``{value: count / total}``.
    ``"mean"``
        The arithmetic mean.
    ``"2ndmom"``
        The second moment about zero, ``E[X^2]``.
    ``"var"``
        The variance, ``E[X^2] - E[X]^2``.
    ``"sdv"``
        The standard deviation.
    ``"med"``
        The median (a half-integer when the sample size is even and the two
        central values differ).
    ``"mode"``
        A list of every value attaining the maximum frequency.

    Parameters
    ----------
    dictionary : dict of {int: int}
        A non-empty frequency distribution ``{value: count}``.

    Returns
    -------
    dict
        The summary described above.

    Examples
    --------
    >>> s = dictionary_statistics({1: 2, 2: 2, 3: 1})
    >>> s["mean"]
    1.8
    >>> s["mode"]
    [1, 2]
    """
    frequencies = dictionary_sort(dictionary)
    relative_frequencies: dict[int, float] = {}
    number_of_objects_counted = 0
    mean = 0.0
    median: float = 0
    mode: list[int] = []
    second_moment = 0.0

    max_frequency = max(frequencies.values())
    for value in frequencies.keys():
        number_of_objects_counted += frequencies[value]
        mean += value * frequencies[value]
        second_moment += (value ** 2) * frequencies[value]
        if frequencies[value] == max_frequency:
            mode.append(value)
    mean = mean / number_of_objects_counted
    second_moment = second_moment / number_of_objects_counted
    variance = second_moment - mean ** 2
    standard_deviation = np.sqrt(variance)

    # Median from cumulative counts, without expanding the data.
    temp_counter = 0
    if number_of_objects_counted % 2 == 1:
        for value in frequencies.keys():
            if temp_counter < number_of_objects_counted / 2:
                temp_counter += frequencies[value]
                if temp_counter > number_of_objects_counted / 2:
                    median = value
    if number_of_objects_counted % 2 == 0:
        for value in frequencies.keys():
            if temp_counter < number_of_objects_counted / 2:
                temp_counter += frequencies[value]
                if temp_counter >= number_of_objects_counted / 2:
                    median = value
        temp_counter = 0
        for value in frequencies.keys():
            if temp_counter < 1 + (number_of_objects_counted / 2):
                temp_counter += frequencies[value]
                if temp_counter >= 1 + (number_of_objects_counted / 2):
                    median = (median + value) / 2

    for value in frequencies.keys():
        relative_frequencies[value] = frequencies[value] / number_of_objects_counted

    return {
        "dist": relative_frequencies,
        "mean": mean,
        "2ndmom": second_moment,
        "var": variance,
        "sdv": standard_deviation,
        "med": median,
        "mode": mode,
    }


def basic_stats(meta_dictionary: MetaDict) -> dict[int, dict[str, Any]]:
    """Apply :func:`dictionary_statistics` across a meta dictionary.

    Parameters
    ----------
    meta_dictionary : MetaDict
        A dictionary mapping each outer key (an ``a`` or a checkpoint ``N``) to
        a frequency distribution.

    Returns
    -------
    dict
        The same keys, each mapped to the full statistics summary of its
        distribution.
    """
    return {
        key: dictionary_statistics(meta_dictionary[key])
        for key in meta_dictionary.keys()
    }


def dists(meta_dictionary: MetaDict) -> dict[int, dict[int, float]]:
    """Return only the relative-frequency distributions of a meta dictionary.

    Equivalent to taking the ``"dist"`` entry of each
    :func:`dictionary_statistics` result.

    Parameters
    ----------
    meta_dictionary : MetaDict
        A dictionary mapping each outer key to a frequency distribution.

    Returns
    -------
    dict
        The same keys, each mapped to a ``{value: relative_frequency}`` dict.
    """
    return {
        key: dictionary_statistics(meta_dictionary[key])["dist"]
        for key in meta_dictionary.keys()
    }


def tabulate(meta_dictionary: MetaDict) -> pd.DataFrame:
    """Render a meta dictionary as a :class:`pandas.DataFrame`.

    Columns are the outer keys and rows are the union of the inner keys, with
    missing cells filled with zeros.  If every value in the table is integral
    the frame is cast to integers; otherwise it is cast to floats.

    Parameters
    ----------
    meta_dictionary : MetaDict
        A dictionary of frequency (or relative-frequency) distributions.

    Returns
    -------
    pandas.DataFrame
        The tabulated data, zero-filled, with an integer or float dtype.
    """
    all_integers = True
    for inner in meta_dictionary.values():
        for value in inner.values():
            if isinstance(value, float):
                all_integers = False
    frame = pd.DataFrame.from_dict(meta_dictionary).fillna(0)
    if all_integers:
        return frame.apply(np.int64)
    return frame.apply(np.float64)

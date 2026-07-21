"""Euclid's Algorithm Analysis.

A small, well-documented library for studying the number of division steps
``T(a, b)`` taken by Euclid's algorithm: implementing the algorithm several
ways, tabulating step-count frequencies over large ranges of integer pairs,
summarising the resulting distributions, and comparing them to the asymptotic
normal behaviour predicted by the theorems of Heilbronn--Porter, Norton, and
Hensley.

The public surface is organised into focused modules:

* :mod:`euclid_analysis.algorithms` -- Euclid's algorithm and variants;
* :mod:`euclid_analysis.fibonacci` -- Fibonacci implementations (worst case);
* :mod:`euclid_analysis.constants` -- the mean/variance constants;
* :mod:`euclid_analysis.frequencies` -- 1-D and 2-D frequency tables;
* :mod:`euclid_analysis.statistics` -- distribution summaries and tabulation;
* :mod:`euclid_analysis.dataio` -- CSV serialisation of the frequency tables;
* :mod:`euclid_analysis.plotting` -- figures and animations (optional, needs
  matplotlib).

The most commonly used names are re-exported here for convenience.
"""

from __future__ import annotations

from . import algorithms, constants, dataio, fibonacci, frequencies, statistics
from .algorithms import (
    count_steps,
    gcd,
    gcd_steps,
    gcdn,
    quotients_remainders,
    remainders,
    write_euclid,
)
from .dataio import DATASETS, load_dataset, load_meta_dict, save_meta_dict
from .fibonacci import binet_fib, fib, fibonacci_list, naive_fib
from .frequencies import dictionary_sort, euclid_alg_frequencies, heilbronn
from .statistics import basic_stats, dictionary_statistics, dists, tabulate

__version__ = "1.0.0"

__all__ = [
    # submodules
    "algorithms",
    "fibonacci",
    "constants",
    "frequencies",
    "statistics",
    "dataio",
    # algorithms
    "gcd",
    "gcdn",
    "count_steps",
    "gcd_steps",
    "remainders",
    "quotients_remainders",
    "write_euclid",
    # fibonacci
    "naive_fib",
    "fibonacci_list",
    "fib",
    "binet_fib",
    # frequencies
    "dictionary_sort",
    "heilbronn",
    "euclid_alg_frequencies",
    # statistics
    "dictionary_statistics",
    "basic_stats",
    "dists",
    "tabulate",
    # dataio
    "load_dataset",
    "load_meta_dict",
    "save_meta_dict",
    "DATASETS",
    "__version__",
]

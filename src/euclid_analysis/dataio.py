"""Serialise frequency meta dictionaries to and from CSV.

The project's precomputed data live in the repository's ``data`` directory as
CSV files with a step count (or gcd value) index and one column per checkpoint.
This module provides a clean round trip between those CSVs and the in-memory
meta-dictionary representation used everywhere else, replacing the repetitive
key-repair blocks of the original single-file script.

A saved table is produced by :func:`euclid_analysis.statistics.tabulate` and
written with :meth:`pandas.DataFrame.to_csv`; on the way back in, the column
headers (which pandas reads as strings) are cast to integers and the
zero-filled cells introduced by tabulation are dropped, recovering the original
sparse meta dictionary.

The named datasets shipped with the repository are enumerated in
:data:`DATASETS`, and :func:`load_dataset` loads them by name.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .statistics import tabulate

__all__ = [
    "DATA_DIR",
    "DATASETS",
    "save_meta_dict",
    "load_meta_dict",
    "load_dataset",
]

FrequencyDict = dict[int, int]
MetaDict = dict[int, FrequencyDict]

#: Absolute path to the repository ``data`` directory (``<repo>/data``),
#: resolved relative to this file so it works regardless of the caller's cwd.
DATA_DIR: Path = Path(__file__).resolve().parents[2] / "data"

#: Human-readable dataset names mapped to their CSV file names.  The keys are
#: the identifiers used throughout the exposition (``B`` and ``C`` are the
#: two-dimensional coprime/all step-count tables; ``H1`` and ``H`` are the
#: one-dimensional coprime/all tables; ``A`` is the large gcd-pair table, which
#: is not shipped because of its size).
DATASETS: dict[str, str] = {
    "B": "euclid_steps_coprime_100001df.csv",
    "C": "euclid_steps_100001df.csv",
    "H1": "euclid_steps_1d-coprime-10001df.csv",
    "H": "euclid_steps_1d-all-10001df.csv",
    "A": "gcd_pairs_100001df.csv",
}


def save_meta_dict(meta_dictionary: MetaDict, path: str | Path) -> None:
    """Tabulate a meta dictionary and write it to ``path`` as CSV.

    Parameters
    ----------
    meta_dictionary : MetaDict
        A dictionary of frequency distributions keyed by ``a`` or checkpoint
        ``N``.
    path : str or pathlib.Path
        Destination CSV path.  Parent directories must already exist.

    See Also
    --------
    load_meta_dict : The inverse operation.
    euclid_analysis.statistics.tabulate : Produces the DataFrame that is saved.
    """
    tabulate(meta_dictionary).to_csv(path)


def load_meta_dict(path: str | Path, *, drop_zeros: bool = True) -> MetaDict:
    """Load a meta dictionary from a CSV written by :func:`save_meta_dict`.

    The CSV's first column is the (integer) inner index -- a step count or a
    gcd value -- and its header row gives the outer keys.  Pandas reads those
    headers as strings, so they are cast back to integers here; the zero cells
    introduced by tabulation are dropped by default to recover the original
    sparse representation.

    Parameters
    ----------
    path : str or pathlib.Path
        Source CSV path.
    drop_zeros : bool, keyword-only, optional
        If ``True`` (default), omit entries whose count is zero, matching the
        sparse dictionaries returned by the frequency routines.  If ``False``,
        keep every cell of the rectangular table.

    Returns
    -------
    MetaDict
        A dictionary mapping each integer outer key to a ``{value: count}``
        frequency dictionary.

    Examples
    --------
    >>> from euclid_analysis.dataio import DATASETS, DATA_DIR
    >>> table = load_meta_dict(DATA_DIR / DATASETS["C"])
    >>> table[1001][5]
    103591
    """
    frame = pd.read_csv(path, index_col=0)
    raw = frame.to_dict(orient="dict")
    result: MetaDict = {}
    for outer_key, column in raw.items():
        outer = int(outer_key)
        if drop_zeros:
            result[outer] = {
                inner: count for inner, count in column.items() if count != 0
            }
        else:
            result[outer] = dict(column)
    return result


def load_dataset(name: str, *, drop_zeros: bool = True) -> MetaDict:
    """Load one of the named datasets in :data:`DATASETS` by key.

    Parameters
    ----------
    name : str
        A key of :data:`DATASETS` (for example ``"C"`` or ``"H1"``).
    drop_zeros : bool, keyword-only, optional
        Passed through to :func:`load_meta_dict`.

    Returns
    -------
    MetaDict
        The loaded meta dictionary.

    Raises
    ------
    KeyError
        If ``name`` is not a known dataset.
    FileNotFoundError
        If the dataset is known but its file is absent (for example the large
        ``"A"`` gcd-pair table, which is not shipped with the repository).
    """
    if name not in DATASETS:
        raise KeyError(
            f"unknown dataset {name!r}; choose from {sorted(DATASETS)}"
        )
    path = DATA_DIR / DATASETS[name]
    if not path.exists():
        raise FileNotFoundError(
            f"dataset {name!r} expected at {path} but the file is not present"
        )
    return load_meta_dict(path, drop_zeros=drop_zeros)

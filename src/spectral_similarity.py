"""Greedy m/z-matched cosine similarity between two MS/MS peak lists.

Simple by design: sort both spectra by m/z, walk them with two pointers,
matching peaks within a fixed Da tolerance. Not the optimal bipartite
matching a library like matchms would do, but a fast, defensible baseline --
exactly the kind of simplicity the official baseline itself favors.
"""
import numpy as np


def cosine_similarity(mzs1, ints1, mzs2, ints2, tolerance_da: float = 0.01) -> float:
    mzs1 = np.asarray(mzs1, dtype=float)
    ints1 = np.asarray(ints1, dtype=float)
    mzs2 = np.asarray(mzs2, dtype=float)
    ints2 = np.asarray(ints2, dtype=float)
    if len(mzs1) == 0 or len(mzs2) == 0:
        return 0.0

    o1 = np.argsort(mzs1)
    o2 = np.argsort(mzs2)
    mzs1, ints1 = mzs1[o1], ints1[o1]
    mzs2, ints2 = mzs2[o2], ints2[o2]

    i = j = 0
    dot = 0.0
    while i < len(mzs1) and j < len(mzs2):
        diff = mzs1[i] - mzs2[j]
        if abs(diff) <= tolerance_da:
            dot += ints1[i] * ints2[j]
            i += 1
            j += 1
        elif diff > 0:
            j += 1
        else:
            i += 1

    norm = np.sqrt(np.sum(ints1**2) * np.sum(ints2**2))
    return float(dot / norm) if norm > 0 else 0.0


def precursor_matches(query_mz: float, ref_mz: float, ppm_tolerance: float = 8.5) -> bool:
    """±ppm precursor mass window -- the community's own tuned value (see RESEARCH.md)."""
    if ref_mz <= 0:
        return False
    ppm_diff = abs(query_mz - ref_mz) / ref_mz * 1e6
    return ppm_diff <= ppm_tolerance

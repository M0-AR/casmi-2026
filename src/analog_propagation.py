"""Mass-shifted analog propagation (Phase 3): score candidate structures by
fingerprint similarity to spectrally-similar "analogs", not just exact
spectral matches. Based on the community's reported scoring function (see
RESEARCH.md):

    AnalogScore(c) = sum_a  Sim(query, a)^4 * Tanimoto(fp_c, fp_a)

Vectorized: Tanimoto against every candidate structure at once via a single
matrix-vector product per analog (intersection = FP_matrix @ analog_fp,
union = popcounts[candidates] + popcount[analog] - intersection), since
looping per-candidate in Python would be far too slow at library scale.
"""
import numpy as np

from fingerprint import hex_to_bits
from spectral_similarity import cosine_similarity


def build_fp_matrix(hex_fingerprints: list[str]) -> np.ndarray:
    """Stacks hex-encoded fingerprints into an (N, n_bits) float32 0/1 matrix."""
    return np.stack([hex_to_bits(h).astype(np.float32) for h in hex_fingerprints])


def analog_scores(
    query_mzs,
    query_intensities,
    analog_candidates: list[dict],  # each: {'mzs', 'intensities', 'fp_row_idx'}
    fp_matrix: np.ndarray,
    fp_popcounts: np.ndarray,
    sim_power: float = 4.0,
    top_k: int = 30,
) -> np.ndarray:
    """Returns an (N,) array of AnalogScore for every row in fp_matrix."""
    sims = np.array(
        [
            cosine_similarity(query_mzs, query_intensities, a["mzs"], a["intensities"])
            for a in analog_candidates
        ]
    )
    order = np.argsort(sims)[::-1][:top_k]

    scores = np.zeros(fp_matrix.shape[0], dtype=np.float64)
    for idx in order:
        sim = sims[idx]
        if sim <= 0:
            continue
        a = analog_candidates[idx]
        analog_fp = fp_matrix[a["fp_row_idx"]]
        intersection = fp_matrix @ analog_fp
        union = fp_popcounts + fp_popcounts[a["fp_row_idx"]] - intersection
        tanimoto = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
        scores += (sim**sim_power) * tanimoto
    return scores

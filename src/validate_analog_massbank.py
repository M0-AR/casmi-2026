"""Validates analog propagation on genuinely hard cases: MassBank structures
with exactly ONE spectrum in the whole library (no exact-duplicate spectrum
exists anywhere), simulating the harder Tier-2/3-style absence of a direct
spectral match. Checks whether fingerprint-similarity-to-spectral-analogs
adds real ranking signal beyond what exact matching alone could do (which
would score ~0 here by construction, since no duplicate spectrum exists).
"""
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from analog_propagation import analog_scores, build_fp_matrix  # noqa: E402

N_QUERIES = 50
MASS_WINDOW_DA = 300.0
SEED = 0


def main():
    t0 = time.time()
    spectra = pd.read_parquet("data/massbank/spectra.parquet")
    structs = pd.read_parquet("data/massbank/structures_with_fingerprints.parquet")
    print(f"{len(spectra):,} spectra, {len(structs):,} structures")

    fp_matrix = build_fp_matrix(structs["morgan_fp_1024_hex"].tolist())
    fp_popcounts = fp_matrix.sum(axis=1)
    ikey_to_row = {ikey: i for i, ikey in enumerate(structs["inchikey"])}
    print(f"fingerprint matrix built: {fp_matrix.shape}, in {time.time()-t0:.1f}s")

    counts = spectra.groupby("inchikey").size()
    singleton_structs = set(counts[counts == 1].index)
    singletons = spectra[spectra.inchikey.isin(singleton_structs)]
    print(f"{len(singletons):,} singleton-structure spectra (no exact duplicate exists)")

    rng = random.Random(SEED)
    query_rows = rng.sample(list(singletons.index), min(N_QUERIES, len(singletons)))

    reciprocal_ranks = []
    for n, qi in enumerate(query_rows):
        query = spectra.loc[qi]
        true_ikey = query.inchikey

        # analog candidate pool: other spectra within a generous mass window,
        # excluding every spectrum of the query's own structure (it's a
        # singleton so that's just this one row, but kept general).
        pool = spectra[
            (spectra.inchikey != true_ikey)
            & (spectra.precursor_mz.sub(query.precursor_mz).abs() <= MASS_WINDOW_DA)
        ]
        analog_candidates = [
            {"mzs": r.mzs, "intensities": r.intensities, "fp_row_idx": ikey_to_row[r.inchikey]}
            for r in pool.itertuples()
            if r.inchikey in ikey_to_row
        ]
        if not analog_candidates:
            reciprocal_ranks.append(0.0)
            continue

        scores = analog_scores(query.mzs, query.intensities, analog_candidates, fp_matrix, fp_popcounts)
        true_row = ikey_to_row[true_ikey]
        rank = int((scores > scores[true_row]).sum()) + 1
        rr = 1.0 / rank if rank <= 25 else 0.0
        reciprocal_ranks.append(rr)
        print(f"  [{n+1}/{len(query_rows)}] pool={len(analog_candidates)} rank={rank} rr={rr:.3f}")

    mrr = float(np.mean(reciprocal_ranks))
    hit_rate = float(np.mean([r > 0 for r in reciprocal_ranks]))
    print(f"\nqueries: {len(reciprocal_ranks)}")
    print(f"MRR@25 (analog propagation only, no exact match possible): {mrr:.4f}")
    print(f"hit rate: {hit_rate:.2%}")
    print(f"done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

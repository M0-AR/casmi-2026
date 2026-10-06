"""Leave-one-out retrieval validation of the Phase 1 approach against the
MassBank reference library (real external data, independent of the
competition's own train.parquet, which is still downloading).

For each held-out query spectrum: remove it from the library, filter
candidates by precursor mass (+-8.5ppm, adduct match), rank candidate
structures by best cosine similarity across their remaining library spectra,
and record the rank of the true structure. Reports MRR@25, matching the
competition's own metric, as an early sanity check that retrieval alone has
real signal before we touch the competition's data.
"""
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from spectral_similarity import cosine_similarity, precursor_matches  # noqa: E402

N_QUERIES = 150
SEED = 0


def main():
    t0 = time.time()
    df = pd.read_parquet("data/massbank/spectra.parquet")
    print(f"{len(df):,} spectra, {df.inchikey.nunique():,} unique structures")

    # Only query structures that have >=2 spectra, so a real library match is possible
    # after removing the query itself.
    counts = df.groupby("inchikey").size()
    eligible_structs = counts[counts >= 2].index
    eligible = df[df.inchikey.isin(eligible_structs)]
    print(f"{len(eligible_structs):,} structures have >=2 spectra (eligible as queries)")

    rng = random.Random(SEED)
    query_idx = rng.sample(list(eligible.index), min(N_QUERIES, len(eligible)))

    records = df.to_dict("records")
    reciprocal_ranks = []

    for qi in query_idx:
        query = df.loc[qi]
        true_inchikey = query.inchikey

        # library = everything except this exact spectrum
        candidates_scores = {}  # inchikey -> best similarity
        for row in records:
            if row["spectrum_id"] == query.spectrum_id:
                continue
            if row["adduct"] != query.adduct:
                continue
            if not precursor_matches(query.precursor_mz, row["precursor_mz"]):
                continue
            sim = cosine_similarity(query.mzs, query.intensities, row["mzs"], row["intensities"])
            prev = candidates_scores.get(row["inchikey"], -1.0)
            if sim > prev:
                candidates_scores[row["inchikey"]] = sim

        ranked = sorted(candidates_scores.items(), key=lambda kv: kv[1], reverse=True)[:25]
        rank = next((i + 1 for i, (ikey, _) in enumerate(ranked) if ikey == true_inchikey), None)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    mrr = float(np.mean(reciprocal_ranks))
    hit_rate = float(np.mean([r > 0 for r in reciprocal_ranks]))
    print(f"\nqueries: {len(reciprocal_ranks)}")
    print(f"MRR@25: {mrr:.4f}")
    print(f"hit rate (true structure anywhere in top 25): {hit_rate:.2%}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

"""Phase 1: exact-match spectral retrieval baseline against the real
competition data -- the first real, gradeable submission.csv.

Streams train.parquet in batches (not loaded fully into memory at once --
the peak-array columns alone are ~6.5GB, which OOM'd this shared,
memory-constrained environment on a first attempt) and, for each batch,
checks every test query against it with vectorized precursor/adduct
filtering before falling back to the per-pair cosine similarity loop.

For each test molecule (which may have 1-16 spectra), pool evidence across
all its spectra: for each spectrum, filter to the same adduct within
+-8.5ppm precursor mass (the community's tuned window -- though see
RESEARCH.md's later finding that this barely matters on timsTOF data),
rank candidate structures by best cosine similarity, then combine across the
molecule's spectra by taking each candidate's single best score from any of
them. Output is deduplicated by OUR canonical InChIKey14 (never the shipped
column -- see RESEARCH.md's canonicalization gotcha), capped at 25,
best-score first.
"""
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "src")
from spectral_similarity import cosine_similarity  # noqa: E402

TRAIN_PATH = "data/train.parquet"
TEST_PATH = "data/test.parquet"
SAMPLE_SUBMISSION_PATH = "data/sample_submission.csv"
STRUCTURES_CANONICAL_PATH = "data/structures_canonical.parquet"
SUBMISSION_PATH = "data/submission_phase1.csv"

PPM_TOLERANCE = 8.5
N_GUESSES = 25
FALLBACK_SMILES = "CCO"
BATCH_SIZE = 131_072  # one parquet row group


def main():
    t0 = time.time()

    print("loading canonical structure table (light)...")
    structs = pd.read_parquet(STRUCTURES_CANONICAL_PATH, columns=["normalized_smiles", "canonical_inchikey14"])
    smiles_to_ikey = dict(zip(structs.normalized_smiles, structs.canonical_inchikey14))
    print(f"  {len(smiles_to_ikey):,} structures, {time.time()-t0:.1f}s")

    test = pd.read_parquet(TEST_PATH)
    print(f"  {len(test):,} test spectra, {test.molecule_id.nunique():,} molecules")

    query_mz = test["precursor_mz"].to_numpy()
    query_adduct = test["adduct"].to_numpy()

    best_by_molecule = defaultdict(dict)  # molecule_id -> {canonical_ikey: (score, smiles)}

    pf = pq.ParquetFile(TRAIN_PATH)
    columns = ["normalized_smiles", "precursor_mz", "adduct", "ms2_mzs", "ms2_normalized_intensities"]
    n_batches = pf.metadata.num_row_groups
    n_rows_seen = 0

    for batch_i, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE, columns=columns)):
        train = batch.to_pandas()
        n_rows_seen += len(train)
        train_mz = train["precursor_mz"].to_numpy()
        train_adduct = train["adduct"].to_numpy()

        for qi, query in test.iterrows():
            ppm_diff = np.abs(train_mz - query_mz[qi]) / query_mz[qi] * 1e6
            mask = (train_adduct == query_adduct[qi]) & (ppm_diff <= PPM_TOLERANCE)
            if not mask.any():
                continue
            candidates = train[mask]
            mid = query.molecule_id
            for _, cand in candidates.iterrows():
                ikey = smiles_to_ikey.get(cand.normalized_smiles)
                if ikey is None:
                    continue
                sim = cosine_similarity(
                    query.ms2_mzs, query.ms2_normalized_intensities,
                    cand.ms2_mzs, cand.ms2_normalized_intensities,
                )
                prev = best_by_molecule[mid].get(ikey, (-1.0, ""))
                if sim > prev[0]:
                    best_by_molecule[mid][ikey] = (sim, cand.normalized_smiles)

        print(
            f"  batch {batch_i+1}/{n_batches} ({n_rows_seen:,} rows seen), "
            f"{time.time()-t0:.1f}s elapsed"
        )

    def top_guesses(molecule_id):
        ranked = sorted(best_by_molecule.get(molecule_id, {}).values(), reverse=True)
        guesses = [smiles for _, smiles in ranked[:N_GUESSES]]
        return ";".join(guesses) if guesses else FALLBACK_SMILES

    submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    submission["smiles"] = submission.molecule_id.apply(top_guesses)
    submission.to_csv(SUBMISSION_PATH, index=False)

    n_guesses = submission.smiles.str.split(";").map(len)
    n_fallback = (submission.smiles == FALLBACK_SMILES).sum()
    print(f"\nwrote {SUBMISSION_PATH}: {len(submission)} rows")
    print(f"guesses per molecule: median {int(n_guesses.median())}, max {n_guesses.max()}")
    print(f"molecules with zero candidates (fallback only): {n_fallback} / {len(submission)}")
    print(f"total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

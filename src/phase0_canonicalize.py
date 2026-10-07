"""Phase 0: build the canonical structure table and verify the
InChIKey14 canonicalization gotcha (see RESEARCH.md) directly against our
own computation, against the full training set.

Does NOT exclude ingest_lib == 'enveda-180': the official baseline notebook
excludes it, but a host reply on the forum (topic 745148) confirms it's a
real, legitimate published spectral library (Zenodo record 21346580), not a
held-out validation set -- and it's actually the single largest ingest_lib
(1.15M of 2.5M spectra, ~45%), not a small held-out slice. Excluding it would
throw away the majority of real training coverage for no demonstrated reason;
the baseline's exclusion looks like it was just a convenient way to shrink a
training sample for a quick tutorial, not a correctness requirement. See
RESEARCH.md for the full correction.

Usage: venv/bin/python src/phase0_canonicalize.py
"""
import sys
import time

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, "src")
from canonicalize import canonical_inchikey14  # noqa: E402

TRAIN_PATH = "data/train.parquet"
OUT_PATH = "data/structures_canonical.parquet"


def main():
    t0 = time.time()
    print("loading train.parquet (normalized_smiles, inchikey14, molecular_formula, ingest_lib)...")
    df = pd.read_parquet(
        TRAIN_PATH,
        columns=["normalized_smiles", "inchikey14", "molecular_formula", "ingest_lib"],
    )
    print(f"  {len(df):,} spectra rows in {time.time() - t0:.1f}s")
    print(f"  by ingest_lib:\n{df.ingest_lib.value_counts().to_string()}")

    structs = df.drop_duplicates(subset=["normalized_smiles"]).reset_index(drop=True)
    print(f"  {len(structs):,} unique structures (by normalized_smiles)")

    tqdm.pandas(desc="canonicalizing")
    structs["canonical_inchikey14"] = structs["normalized_smiles"].progress_apply(canonical_inchikey14)

    failed = structs["canonical_inchikey14"].isna().sum()
    print(f"\n{failed:,} / {len(structs):,} structures failed to canonicalize ({failed / len(structs):.2%})")

    valid = structs.dropna(subset=["canonical_inchikey14"])
    mismatches = (valid["canonical_inchikey14"] != valid["inchikey14"]).sum()
    print(
        f"shipped vs. canonical InChIKey14 mismatch: {mismatches:,} / {len(valid):,} "
        f"({mismatches / len(valid):.2%}) -- forum reported ~4.4%"
    )

    n_shipped_distinct = valid["inchikey14"].nunique()
    n_canonical_distinct = valid["canonical_inchikey14"].nunique()
    print(
        f"distinct keys: {n_shipped_distinct:,} shipped -> {n_canonical_distinct:,} canonical "
        f"({n_shipped_distinct - n_canonical_distinct:,} collapsed, "
        f"forum reported 66490 -> 63449, ~1/22)"
    )

    structs.to_parquet(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH} ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()

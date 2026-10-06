"""Phase 0: build the canonical structure table and verify the
InChIKey14 canonicalization gotcha (see RESEARCH.md) directly against our
own computation, against the full training set.

Excludes ingest_lib == 'enveda-180' (Enveda's own held-out validation set),
matching the official baseline's own choice to keep it separate.

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
        filters=[("ingest_lib", "!=", "enveda-180")],
    )
    print(f"  {len(df):,} spectra rows in {time.time() - t0:.1f}s")

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

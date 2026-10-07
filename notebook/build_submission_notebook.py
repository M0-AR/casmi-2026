"""Builds the real Kaggle submission notebook from plain Python cell sources,
avoiding hand-escaped JSON. Mirrors the project's own canonicalize.py /
spectral_similarity.py / phase1_retrieval.py logic, adapted for Kaggle's
constraints (see RESEARCH.md / FINDINGS.md for why each piece is needed):

  - rdkit isn't preinstalled and can't be pip-installed at scoring time
    (internet disabled). Fix: unzip the community's offline wheel
    (prvsiyan/rdkit-wheel-offline, exact pinned version 2026.3.3) directly
    with zipfile (no pip needed -- python3.12 has no pip at all on this
    image) and run rdkit-dependent code via a /usr/bin/python3.12 subprocess
    with PYTHONPATH pointed at the extracted package (the kernel's own
    Python is 3.13, ABI-incompatible with the cp312 wheel).
  - train.parquet's peak-array columns are ~6.5GB in memory -- too much for
    this and Kaggle's own session limits. Fix: stream it in row-group
    batches via pyarrow, never materializing the whole thing at once.

Usage: python notebook/build_submission_notebook.py
"""
import json

CELLS = []


def cell(source: str):
    CELLS.append(source.strip() + "\n")


cell("""
import os
print('--- /kaggle/input ---')
for root, dirs, files in os.walk('/kaggle/input'):
    print(root, len(files), 'files')
""")

cell("""
# rdkit isn't preinstalled and can't be pip-installed with internet disabled.
# The offline wheel's own python3.12 has no pip either, but a .whl is just a
# zip file -- unzip it directly and run rdkit code via a python3.12
# subprocess (the kernel's own Python is 3.13, ABI-incompatible with the
# wheel's cp312 build).
import glob
import zipfile

PKGS_DIR = '/kaggle/working/rdkit_pkgs'
os.makedirs(PKGS_DIR, exist_ok=True)
# Mount path confirmed by direct observation on this kernel (varied from an
# earlier project's finding -- don't assume, always check the real listing
# above): /kaggle/input/datasets/<owner>/<slug>/, not /kaggle/input/<slug>/.
wheel_candidates = (
    glob.glob('/kaggle/input/datasets/prvsiyan/rdkit-wheel-offline/*.whl')
    + glob.glob('/kaggle/input/rdkit-wheel-offline/*.whl')
)
assert wheel_candidates, 'rdkit wheel not found at either known mount path'
wheel = wheel_candidates[0]
print('using wheel:', wheel)
with zipfile.ZipFile(wheel) as zf:
    zf.extractall(PKGS_DIR)
print('extracted:', os.listdir(PKGS_DIR))
""")

# The helper script is embedded as base64 (not a nested Python string) --
# three levels of string-nesting (this generator -> the cell source -> the
# written .py file's own escape sequences) silently mis-escaped "\n" into a
# literal newline on a real Kaggle run and broke the helper's own syntax.
# Base64 has no characters that need escaping at any level, so it can't
# suffer the same failure mode.
import base64

with open("notebook/canonicalize_helper.py", "rb") as f:
    _helper_b64 = base64.b64encode(f.read()).decode("ascii")

cell(f'''
import base64

with open('/kaggle/working/canonicalize_helper.py', 'wb') as f:
    f.write(base64.b64decode("{_helper_b64}"))
''')

cell("""
# Phase 0: canonicalize every unique training structure (see RESEARCH.md --
# the shipped inchikey14 column isn't tautomer-canonicalized the way the
# real scorer is, so trusting it would silently cost MRR).
import subprocess
import time

import pandas as pd

COMP_DIR = '/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra'
TRAIN_PATH = f'{COMP_DIR}/train.parquet'
TEST_PATH = f'{COMP_DIR}/test.parquet'
SAMPLE_SUBMISSION_PATH = f'{COMP_DIR}/sample_submission.csv'

t0 = time.time()
unique_smiles = pd.read_parquet(TRAIN_PATH, columns=['normalized_smiles'])['normalized_smiles'].unique().tolist()
print(f'{len(unique_smiles):,} unique structures, {time.time()-t0:.1f}s')

with open('/kaggle/working/smiles_in.txt', 'w') as f:
    f.write('\\n'.join(unique_smiles))

env = os.environ.copy()
env['PYTHONPATH'] = PKGS_DIR
r = subprocess.run(
    ['/usr/bin/python3.12', '/kaggle/working/canonicalize_helper.py',
     '/kaggle/working/smiles_in.txt', '/kaggle/working/ikeys_out.txt'],
    capture_output=True, text=True, env=env,
)
print('canonicalize stdout:', r.stdout[-1000:])
print('canonicalize stderr:', r.stderr[-1000:])
print('canonicalize returncode:', r.returncode)
assert r.returncode == 0, 'canonicalization subprocess failed'

with open('/kaggle/working/ikeys_out.txt') as f:
    canonical_keys = f.read().split('\\n')
assert len(canonical_keys) == len(unique_smiles)
smiles_to_ikey = dict(zip(unique_smiles, canonical_keys))
n_failed = sum(1 for v in smiles_to_ikey.values() if v == '')
print(f'canonicalization done: {n_failed} / {len(smiles_to_ikey)} failed to parse, {time.time()-t0:.1f}s total')
""")

cell('''
# Cosine similarity between two MS/MS peak lists (greedy m/z-matched, see
# spectral_similarity.py in the repo for the full rationale) and the
# precursor-mass ppm filter.
import numpy as np


def cosine_similarity(mzs1, ints1, mzs2, ints2, tolerance_da=0.01):
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
''')

cell("""
# Phase 1: exact-match retrieval, streaming train.parquet in batches (the
# peak-array columns are ~6.5GB in memory if loaded all at once -- see
# FINDINGS.md). For each test molecule (which may have several spectra),
# pool evidence across all its spectra: filter to the same adduct within
# +-8.5ppm precursor mass, rank candidates by best cosine similarity, and
# combine each candidate's single best score from any of the molecule's
# spectra. Output deduplicated by OUR canonical InChIKey14, capped at 25.
import pyarrow.parquet as pq
from collections import defaultdict

PPM_TOLERANCE = 8.5
N_GUESSES = 25
FALLBACK_SMILES = 'CCO'
BATCH_SIZE = 131_072

t0 = time.time()
test = pd.read_parquet(TEST_PATH)
print(f'{len(test):,} test spectra, {test.molecule_id.nunique():,} molecules')

query_mz = test['precursor_mz'].to_numpy()
query_adduct = test['adduct'].to_numpy()

best_by_molecule = defaultdict(dict)

pf = pq.ParquetFile(TRAIN_PATH)
columns = ['normalized_smiles', 'precursor_mz', 'adduct', 'ms2_mzs', 'ms2_normalized_intensities']
n_batches = pf.metadata.num_row_groups
n_rows_seen = 0

for batch_i, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE, columns=columns)):
    train = batch.to_pandas()
    n_rows_seen += len(train)
    train_mz = train['precursor_mz'].to_numpy()
    train_adduct = train['adduct'].to_numpy()

    for qi, query in test.iterrows():
        ppm_diff = np.abs(train_mz - query_mz[qi]) / query_mz[qi] * 1e6
        mask = (train_adduct == query_adduct[qi]) & (ppm_diff <= PPM_TOLERANCE)
        if not mask.any():
            continue
        candidates = train[mask]
        mid = query.molecule_id
        for _, cand in candidates.iterrows():
            ikey = smiles_to_ikey.get(cand.normalized_smiles)
            if not ikey:
                continue
            sim = cosine_similarity(
                query.ms2_mzs, query.ms2_normalized_intensities,
                cand.ms2_mzs, cand.ms2_normalized_intensities,
            )
            prev = best_by_molecule[mid].get(ikey, (-1.0, ''))
            if sim > prev[0]:
                best_by_molecule[mid][ikey] = (sim, cand.normalized_smiles)

    print(f'  batch {batch_i+1}/{n_batches} ({n_rows_seen:,} rows seen), {time.time()-t0:.1f}s elapsed')
""")

cell("""
# Write submission.csv -- starting from the sample submission guarantees the
# right molecule_ids, in the right order.
def top_guesses(molecule_id):
    ranked = sorted(best_by_molecule.get(molecule_id, {}).values(), reverse=True)
    guesses = [smiles for _, smiles in ranked[:N_GUESSES]]
    return ';'.join(guesses) if guesses else FALLBACK_SMILES

submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)
submission['smiles'] = submission.molecule_id.apply(top_guesses)
submission.to_csv('/kaggle/working/submission.csv', index=False)

n_guesses = submission.smiles.str.split(';').map(len)
n_fallback = (submission.smiles == FALLBACK_SMILES).sum()
print(f'wrote submission.csv: {len(submission)} rows')
print(f'guesses per molecule: median {int(n_guesses.median())}, max {n_guesses.max()}')
print(f'molecules with zero candidates (fallback only): {n_fallback} / {len(submission)}')
print(f'total time: {time.time()-t0:.1f}s')
""")

notebook = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source,
        }
        for source in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT_PATH = "/tmp/claude-1000/-home-md-src/a6a81434-6a0f-4812-b44e-1b9b1d337e66/scratchpad/casmi-submission/submission.ipynb"
with open(OUT_PATH, "w") as f:
    json.dump(notebook, f)
print(f"wrote {OUT_PATH}, {len(CELLS)} cells")

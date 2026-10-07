# CASMI 2026 Agent

An entry for [Enveda CASMI 2026 — Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)
($50,000 prize pool): given 1-16 tandem mass spectra (MS/MS) for an unknown
molecule, predict its 2D chemical structure as a ranked list of up to 25
SMILES candidates, scored by Mean Reciprocal Rank @ 25 against the real
hidden test set.

## What this is

A from-scratch retrieval pipeline, built and validated incrementally against
real data at every stage — nothing here is assumed without direct evidence.
See [`RESEARCH.md`](RESEARCH.md) for the full competitive-landscape
investigation (what the host's own baseline does, what's actually winning,
a confirmed data leak in the public test file, and several hard-won
community lessons about where the metric's remaining skill ceiling actually
is) and [`PLAN.md`](PLAN.md) for the phased build plan those findings drive.

## Status

**Phase 0 and Phase 1 complete and run successfully end-to-end on real
Kaggle infrastructure** (not just locally):

- **Phase 0** — canonicalizes every one of the real training set's 277,566
  unique structures to InChIKey14 the exact way the competition's own scorer
  does (RDKit tautomer canonicalization). This matters because the shipped
  `inchikey14` column in the data is *not* canonicalized this way — trusting
  it silently costs score on deduplication and matching. Confirmed via three
  independent measurements (two external structure sets plus the real
  training data itself) that this is a real, reproducible gotcha, not a
  one-off artifact.
- **Phase 1** — exact-match spectral retrieval: for each test molecule,
  pools evidence across all its spectra, filters the full training set by
  precursor mass (±8.5ppm) and adduct, ranks candidates by cosine spectral
  similarity, and writes a valid, deduplicated top-25 submission. Validated
  on real data: 400/400 molecules got real candidates, 0 fallbacks, every
  one of 10,000 candidate SMILES parses correctly.
- Also validated the Phase 3 (analog propagation) mechanism independently
  against an external 139k-spectrum reference library before the
  competition's own 3GB training set had finished downloading — see
  `src/validate_analog_massbank.py` / `validate_retrieval_massbank.py`.

**Not yet submitted to the competition leaderboard** — Kaggle requires
identity verification to submit to a cash-prize competition, which isn't
available on this account. The pipeline itself is complete, real, and
produces a valid `submission.csv`; see `notebook/build_submission_notebook.py`
for the exact, working Kaggle notebook that proves it end-to-end (full run
log: 277,566 structures canonicalized with 0 failures, retrieval against the
complete 2.5M-spectrum training set, valid output, ~82 minutes total — well
inside the competition's 9-hour runtime limit).

**Remaining phases** (2 — local CV harness, 3 — expand retrieval coverage,
4 — reranker, 5 — the genuinely unsolved novel-structure tier) are planned
in detail in `PLAN.md` but not yet built.

## Repo layout

```
RESEARCH.md                   competitive-landscape investigation: what's
                                actually winning, a confirmed data leak,
                                hard-won community lessons, and the
                                InChIKey14 canonicalization gotcha
PLAN.md                        the phased build plan those findings drive,
                                with a concrete verification step per phase
src/
  canonicalize.py               RDKit tautomer canonicalization matching
                                 the real scorer (reproduces its own worked
                                 glucose example exactly)
  fingerprint.py                hex-encoded bit fingerprint decode + Tanimoto
  spectral_similarity.py        greedy m/z-matched cosine similarity +
                                 precursor-mass ppm filter
  analog_propagation.py         vectorized AnalogScore (fingerprint Tanimoto
                                 weighted by spectral similarity to analogs)
  phase0_canonicalize.py        canonicalizes every unique real training
                                 structure, verifies the mismatch-rate gotcha
  phase1_retrieval.py           exact-match retrieval baseline, streamed in
                                 batches (train.parquet's peak-array columns
                                 are ~6.5GB -- too much to load at once)
  validate_retrieval_massbank.py / validate_analog_massbank.py
                                 validate the retrieval/analog mechanisms
                                 against an external reference library,
                                 independent of the competition's own data
notebook/
  build_submission_notebook.py  generates the real Kaggle submission
                                 notebook programmatically (handles two
                                 real Kaggle-specific constraints: rdkit
                                 isn't preinstalled and can't be
                                 pip-installed with internet disabled, and
                                 a nested-string-escaping bug that silently
                                 broke on a real run until fixed)
  canonicalize_helper.py        the rdkit-dependent half of Phase 0, run via
                                 a python3.12 subprocess bridge (the
                                 kernel's own Python is 3.13, incompatible
                                 with the offline rdkit wheel's cp312 build)
```

## Running it

```bash
python3 -m venv venv
./venv/bin/pip install rdkit==2026.3.3 pandas pyarrow scikit-learn tqdm
```

Local scripts (`src/phase0_canonicalize.py`, `src/phase1_retrieval.py`)
expect the competition's `train.parquet`/`test.parquet`/`sample_submission.csv`
in `data/` (not checked in — see the competition page to download them after
accepting its rules).

To regenerate the actual Kaggle submission notebook from source:

```bash
python3 notebook/build_submission_notebook.py
```

## License

MIT — see [LICENSE](LICENSE).

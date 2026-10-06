# CASMI 2026 — competitive research

Research done directly against the live competition (Kaggle leaderboard, forum
threads, and public notebooks/datasets, pulled via the Kaggle API) on
2026-10-06, before writing any pipeline code.

## The task, precisely

Given 1-16 MS/MS spectra for an unknown molecule (grouped by `molecule_id`),
predict its 2D structure as a ranked list of up to 25 SMILES, best guess
first. Scored by Mean Reciprocal Rank @ 25 (MRR@25): a hit at rank 1 scores
1.0, rank 2 scores 0.5, ..., rank 25 scores 0.04, absent scores 0. Matching is
**not** exact-string: both your SMILES and the true answer are passed through
RDKit tautomer canonicalization (pinned 2026.03.3) and reduced to the first
block of their InChIKey (InChIKey14) before comparing, so stereochemistry and
tautomer form are never penalized.

Hidden test molecules fall into 3 novelty tiers (distribution not disclosed):
1. **In public spectral libraries** — findable by spectral similarity.
2. **Known structure, no public spectra** — in PubChem/COCONUT, reachable by
   database lookup, not spectral matching.
3. **Genuinely novel** — not in any database, must be predicted de novo.

## Real data, verified directly (not just from docs)

- `test.parquet`: 1,213 spectra / 400 molecules in the public (placeholder)
  copy. Columns: `molecule_id, spectrum_id, ms2_mzs, ms2_normalized_intensities,
  base_peak_intensity, adduct, ionization_mode, instrument_type, precursor_mz,
  collision_energy_orig, collision_energy_ev, collision_energy_orig_units`.
- `train.parquet`: ~2.5M spectra / ~275k structures, 3GB. 18 columns per the
  official baseline notebook's own code: `ingest_lib, normalized_smiles,
  inchikey, inchikey14, molecular_formula, ionization_mode, instrument_type,
  adduct, adduct_orig, precursor_mz, precursor_error_ppm, ms2_mzs,
  ms2_normalized_intensities, num_peaks, base_peak_intensity,
  collision_energy_ev, collision_energy_orig, collision_energy_orig_units`.
  **No `molecule_id`/`spectrum_id`** — intentionally excluded (confirmed by
  the host's own baseline code and a forum thread); group training spectra by
  `inchikey14` instead.
- `sample_submission.csv`: `molecule_id,smiles` with 25 semicolon-joined
  placeholder SMILES (`CCO` repeated) per row, 400 rows.
- `ingest_lib` has a special value `'enveda-180'` — Enveda's own held-out
  validation set, which the official baseline explicitly excludes from
  training (`filters=[('ingest_lib', '!=', 'enveda-180')]`). Worth treating
  the same way: don't train on it, it's closer to a clean internal benchmark.

## Critical correctness gotcha (verified via forum + will re-verify ourselves)

The shipped `inchikey14` column in `train.parquet` is **not** RDKit-tautomer-
canonicalized, but the actual scorer **is**. A community member who
canonicalized all 66,490 distinct shipped structures found 2,904 (~4.4%) got a
different key after canonicalization, and the 66,490 shipped-distinct
structures collapse to 63,449 canonically-distinct ones (~1 in 22 has an
unrecognized "twin"). Practical effect: deduplicating candidates on the
shipped column wastes submission slots on molecules the scorer already
counted as hit, and matching retrieved library structures against it can miss
a correct answer filed under a different tautomer spelling. **We must
recompute the canonical InChIKey14 ourselves (RDKit 2026.03.3, tautomer
canonicalization) for every structure we handle** — grouping, deduplication,
and matching all depend on this, not the shipped column.

Other flagged-but-unconfirmed data quirks worth checking ourselves before
relying on them: a possible one-electron-mass offset in positive-mode m/z
(forum topic 743395, 3 votes, unresolved).

## What's actually winning (community state, 2026-10-06)

**The official host baseline is a pure de novo generator and nobody
competitive uses it.** `inversion/casmi-denovo-tutorial-notebook` (281 votes,
the single most-upvoted notebook) trains a from-scratch encoder-decoder
Transformer: sinusoidal (m/z, intensity) peak embeddings -> 6-layer
transformer encoder -> 6-layer transformer decoder generating BPE-tokenized
SMILES via temperature sampling (25 samples/molecule at inference),
deduplicated on (uncanonicalized) InChIKey14. The community's own assessment,
directly from the forum ("Why no one is using encoder-decoder models?",
RISHAV KUMAR): generated SMILES are frequently invalid, it's slow, and "all
the shared code is around some ranker model" instead.

**The real, legitimate ceiling right now is ~0.339-0.350+ MRR@25**, from a
detailed, openly-shared pipeline (haideptry, Kaggle EXPERT tier, posted as a
collaborative "post-mortem" inviting feedback — this is not a locked solution,
it's explicitly meant to be built on). Its 4-channel design:

1. **Exact spectral library matching** — precursor mass filtering + spectral
   entropy similarity + cosine similarity + reference peak matching. Strong
   whenever the compound is already in a reference library (Tier 1).
2. **Mass-shifted analog propagation** — for Tier 2, search spectrally-similar
   "analogs" within a mass-shift window and propagate structural evidence
   through known, interpretable transformations (±CH2, ±OH, ±Hexose, etc.),
   scored as `AnalogScore(c) = Σ_a Sim(q,a)^4 · Tanimoto(f_c, f_a)`.
3. **Chemical/substructure heuristics** ("MetFrag-lite") — approximate bond-
   dissociation reasoning, neutral-loss matching, substructure consistency;
   used as a candidate filter/prior, not a full solver.
4. **Spectrum-to-fingerprint transformer** ("FPNet") — a small 6-layer
   transformer predicting a 6,930-bit fingerprint distribution from the
   spectrum; scored against each candidate's real fingerprint via a fast
   vectorized Bernoulli log-likelihood (`f_c · z`), so tens of thousands of
   candidates can be scored cheaply.

All four channels' outputs feed ~30 engineered features into a **HistGBM
ensemble reranker** (4 seeds, 2 calibration settings) with confidence-based
gating -> top-25 output. This matches a second, independent community thread
("All you need is a good ranker", 21 votes): once candidates are retrieved,
the reranker is reported as the actual remaining bottleneck, not fancier
candidate generation — CV MRR was 0.584 in one "regime" vs. 0.846 in an
easier one, with very few true structures missing from the top-25 pool
entirely; the gap is in ranking order, not recall.

Concrete tuning findings from the same source, worth adopting directly:
- A **tight ±8.5ppm precursor mass window** beat wider windows (too many
  decoys hurt ranking of harder classes).
- **Curated natural-product databases (COCONUT 2.0, ChEBI, LipidMaps)**
  outperformed blind PubChem isomer expansion as a Tier-2/3 candidate source.
- At least one other team reported **Random Forest** beating other ranker
  choices, worth comparing against HistGBM rather than assuming the latter
  is strictly better.

## The leaderboard is currently unreliable — do not calibrate against it

The current public top scores (~0.44-0.47) are **not legitimate skill** — a
confirmed, still-unpatched data leak (forum topic, 18 votes, most recent
comment this morning) exists because the public `test.parquet` is literally
built from training spectra, so for a meaningful share of Tier-1 molecules
the leaked placeholder structure IS the correct answer. A public notebook
(`imranarif536/casmi26-v44-pairtail-locked-top1`) exploits this directly and
has been forked by dozens of teams. As of this research, hosts have not
confirmed whether the final private rerun will remap/shuffle the IDs. Do not
build anything that depends on this, and do not read today's public LB
ranking as real signal — the legitimate reference point is haideptry's
openly-documented 0.339-0.350, not the leaderboard's current top.

## External resources worth pulling in

- **`samartalwar/casmi-2026-spectral-library-massbankharmonized`** (Kaggle
  Dataset) — 139,744 spectra / 20,204 unique structures from public MassBank,
  already harmonized to this competition's own column schema, with
  precomputed Morgan + MACCS fingerprints, including 648 real spectra from
  the actual historical CASMI 2012/2016 challenges. A direct, ready-to-attach
  second reference library, complementary to Enveda's own natural-product-
  skewed `train.parquet` (broader general chemical space, not just NPs).
- **COCONUT 2.0 / ChEBI / LipidMaps** — natural-product/curated structure
  databases, reported by top teams as the right Tier-2/3 candidate source
  (vs. blind PubChem expansion, which hurt).
- **CFM-ID 4 pretrained models / MassSpecGym open weights** — both under live
  forum discussion for prize eligibility; not yet resolved as of this
  research. Do not build a dependency on either without confirming license
  compliance first, given the data's own CC BY-NC 4.0 restriction already in
  force.

## Binding constraints (from the rules page, verified directly)

- Code competition, notebook-only submissions, ≤9h runtime (CPU or GPU),
  internet disabled at scoring time. Freely & publicly available external
  data/pretrained models are allowed if attached as Kaggle Datasets beforehand.
- Max team size 5, 5 submissions/day, 2 final submissions selected for judging.
- Prizes: $16k / $12k / $9k / $7k / $6k (top 5). **Winner license: MIT. Data
  license: CC BY-NC 4.0 (non-commercial only)** — binding on any external data
  or pretrained weights used too.
- Entry deadline 2026-12-07, final submission deadline 2026-12-14.

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
- `ingest_lib` has 11 distinct values. Verified directly against the full,
  real `train.parquet` (2,539,608 spectra / 277,566 structures -- matches
  the docs' "~2.5m spectra" / "~275k structures" closely): `enveda-180`
  (1,153,785 spectra -- **the single largest library, ~45% of all data**),
  `pluskal_ms2` (527,581), `riken` (347,171), `gnps` (220,849), `massbank`
  (101,727), `mona` (92,416), `spectraverse` (50,933), `msdial` (40,765),
  `drug_plus` (2,545), `enveda-np-examples` (1,184 -- this is the small
  natural-product panel top teams reference, e.g. "241 enveda-np-examples
  molecules" in the "~30 submissions" thread above), `masaryk` (652).

  **Correction to an earlier assumption in this doc**: the official baseline
  notebook excludes `ingest_lib == 'enveda-180'`
  (`filters=[('ingest_lib', '!=', 'enveda-180')]`), which we initially read
  as "Enveda's own held-out validation set, keep separate." A host reply on
  the forum (topic 745148) confirms this is wrong -- Enveda-180 is a real,
  legitimate **published spectral library** (Zenodo record 21346580), fine
  to use, not a held-out set, and as the counts above show it's actually the
  *majority* of the real training data, not a small slice. The baseline's
  exclusion looks like it was just a convenient way to shrink the training
  sample for a quick single-session tutorial (consistent with its separate
  `MAX_TRAIN_SPECTRA = 200_000` cap), not a correctness requirement. **We do
  not exclude it** -- `src/phase0_canonicalize.py` was updated accordingly.

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

**Independently verified** (2026-10-06): ran our own RDKit 2026.3.3 tautomer
canonicalization (`src/canonicalize.py`) against a completely separate
20,204-structure set (the MassBank reference library, see below) — 0 parse
failures, and a **4.07% shipped-vs-canonical mismatch rate**, closely
matching the forum's 4.4% figure on the competition's own structures. Two
independent structure sets landing in the same ~4% range confirms this is a
real, stable phenomenon (not a one-off artifact) and that our canonicalizer
is implemented correctly — it also reproduces the competition's own worked
glucose example (`docs/Evaluation` page) exactly.

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

## The real bottleneck, with numbers (2026-10-06 update)

A separate forum post ("Lessons from ~30 submissions: where the MRR actually
goes", 11 votes) and the detailed exchange underneath it are the single most
substantive, statistically rigorous source found so far -- a team reporting
honest ablations with t-values and held-out validation, not just a final
score. Several findings **refine or directly contradict** earlier entries in
this document; noted below.

**The central finding: once a candidate pool contains the truth, errors are
overwhelmingly isomer errors, not formula/mass errors.** When the true
structure was in their candidate list but not ranked first, the wrong winner
shared the exact same molecular formula **~98% of the time**. This reframes
the whole problem: mass-window tuning and formula filtering are not where the
remaining skill is -- separating constitutional isomers of the same formula
is. They built a "same-formula isomer panel" (truth + its best-scoring
same-formula rivals) as a far more sensitive dev metric than overall MRR for
exactly this reason.

**Correction to the ±8.5ppm mass-window finding above**: on *instrument-
matched* timsTOF spectra (which is 100% of this competition's test set),
99% of precursor masses fall within ~5ppm, so widening 8.5->10ppm barely
changes candidate counts at all -- the window is already full of exact-mass
isomers. Mass window width is not a meaningful lever here; don't spend time
tuning it.

**Contradicts the earlier "curated NP databases > blind PubChem" finding**:
this team found ChEBI/LipidMaps/NPAtlas added *no* coverage beyond PubChem
union COCONUT for their missing molecules, and that curated-database
membership flags looked great on their own holdout but hurt on the real
leaderboard (their holdout molecules were over-represented in curated DBs --
a holdout-construction artifact, not a real signal). **Open disagreement
between two independent top teams** -- worth testing both claims ourselves
rather than trusting either.

**Rigorous example of noise discipline, worth copying exactly**: another team
in the same thread measured an in-silico-fragmentation improvement at +0.065
MRR (t=4.6) on their tuning panel -- but after re-validating on four
*independent, held-out* panels (gnps/riken/mona+msdial/massbank), the real
pooled effect was +0.023 ± 0.007 (t=3.45, n=1000), a **2.8x inflation** from
having tuned hyperparameters on the same panel used to measure the gain. This
smaller, honest number exactly predicted their real leaderboard result (no
visible change: 0.341 -> 0.336, within the ~0.016 noise floor for ~130 public
molecules). **Lesson for our own Phase 2 CV harness: a gain must hold on a
panel the tuning never touched before it's trusted.**

**Other concrete, actionable lessons from the same thread:**
- Don't analyze the visible `test.parquet` at all -- it's a placeholder
  drawn from training data (see the leak section above); any statistic
  computed on it is meaningless for the hidden rerun.
- Their own tautomer-canonicalization check found **~8% of molecules had a
  tautomer copy of the truth under a different key** on their holdout --
  same phenomenon as our own 4-4.4% finding on two other structure sets,
  different population, same direction. Strengthens the case this is real
  and worth guarding against everywhere, not a fluke of one dataset.
- Noise floor: ~0.0025 MRR per test molecule; reseeding the same GBDT
  reranker moved their holdout MRR by up to ±0.007. Don't chase CV
  differences below ~0.005 (ties directly into Phase 2's plan).
- "Novel molecule" simulations (deleting a known structure and checking if a
  generator recovers it) overstated real reach ~3x vs. genuinely-absent
  molecules -- validate Phase 5 only against real gaps, not simulated ones.
- Once formula is fixed, similarity signals that don't encode atom
  connectivity (e.g. simple fingerprint similarity) plateau quickly --
  structure-dependent evidence (in-silico fragmentation parsimony, not
  hand-written bond-breaking chemistry rules, which this team found added
  nothing) is where real isomer-separating gains come from. Concrete,
  ablated fragmentation-scorer recipe that worked for them: down-weight
  2-bond cleavages (x0.6 vs. 1-bond), linear (not sqrt) intensity weighting,
  tight 0.005 Da tolerance, H-shifts -2..+3.

**Implication for PLAN.md**: Phase 4 (reranker) needs to prioritize
isomer-discriminating features (in-silico fragmentation parsimony scoring)
over additional formula/mass-window tuning, and Phase 2's CV harness must
validate every change against a panel it wasn't tuned on before trusting the
number -- added as explicit plan updates.

## Own validation: analog propagation alone is weak but real (2026-10-06)

Built `src/analog_propagation.py` (vectorized AnalogScore, see RESEARCH.md's
4-channel description above) and tested it on the genuinely hardest MassBank
cases: the 8,651 structures with **zero duplicate spectra anywhere** in the
139,744-spectrum library, so exact matching cannot possibly work and any
score has to come from fingerprint similarity to spectrally-similar-but-
different structures. Result on 50 such queries, no formula restriction, a
loose ±300 Da mass window, candidate pools of 10k-105k structures: **MRR@25 =
0.0117, 6.00% hit rate**. Far below Phase 1's exact-match 0.73 on easy cases
(expected -- this is deliberately the hardest slice), but 35-250x above
random-guess chance given the pool sizes involved (expected chance hit rate
for a pool of 100k is ~0.025%). Confirms analog propagation alone carries
real, non-trivial signal, but is weak as a standalone ranker -- consistent
with the community's own framing that it's one reranker input feature among
several, not a complete answer, and with Tier-3-style absence being the part
of this problem nobody has solved well yet.

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

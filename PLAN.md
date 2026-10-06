# CASMI 2026 — plan

Builds directly on [RESEARCH.md](RESEARCH.md). Each phase has a concrete,
checkable goal — no phase is "done" until its verification step passes
against real data, not just until code runs.

## Phase 0 — correctness infrastructure (no model yet)

**Goal:** a canonical-InChIKey14 pipeline we trust, because every later
phase's correctness depends on it (see RESEARCH.md's canonicalization gotcha).

- Load `train.parquet`, recompute RDKit-tautomer-canonicalized InChIKey14 for
  every distinct `normalized_smiles` (2026.03.3, matching the scorer exactly).
- Build the grouped structure: `inchikey14 -> {smiles, formula, all spectra}`.
- Implement the submission writer: dedup candidates by *our* canonical key
  (never the shipped one), cap at 25, fallback to a valid placeholder if a
  molecule gets zero candidates.
- **Verify:** spot-check against the ~4.4%-mismatch finding — our
  canonicalization should disagree with the shipped `inchikey14` column on a
  similar fraction of structures. If it doesn't, something's wrong with our
  canonicalization, not the shipped data.

## Phase 1 — exact-match retrieval baseline (first real, gradeable submission)

**Goal:** a valid submission.csv that scores above 0 via real spectral
matching — proves the whole pipeline end-to-end before adding complexity.

- Precursor mass filter (±8.5ppm, per the community's tuning finding) +
  adduct match against `train.parquet` as the reference library.
- Rank candidates by spectral similarity (cosine or entropy similarity) on
  normalized peak lists.
- Pool across a molecule's spectra (a molecule may have 1-16), since scoring
  is per-molecule not per-spectrum.
- **Verify:** submit to Kaggle. This is intentionally the first submission —
  confirms format acceptance and gives a real (if modest) baseline score,
  rather than finding out about a pipeline bug on a later, more complex
  submission.

## Phase 2 — local CV harness (before spending any more submissions)

**Goal:** stop relying on the leaderboard (unreliable right now — see
RESEARCH.md) or burning the 1-submission/day budget to measure progress.

- Hold out a chunk of `train.parquet`'s own molecules, split by canonical
  `inchikey14` (never by spectrum — a molecule's spectra must stay together,
  same mistake the official baseline's own code was careful to avoid).
- Compute MRR@25 locally the same way the real scorer does (canonical
  InChIKey14 match).
- Build a **second, independent holdout panel** from a different source
  (e.g. the MassBank library) in addition to the train.parquet-derived one —
  a hard lesson from RESEARCH.md's "~30 submissions" thread is that a change
  can look real on one panel and vanish (or reverse) on another; two
  disagreeing proxies is itself useful information, not a bug to resolve away.
- **Verify:** local CV score is stable across random seeds/splits (expect
  ~±0.007 noise from reranker reseeding alone, per the same thread) before
  trusting it to guide Phase 3+ decisions. Never tune a change's
  hyperparameters on the same panel used to report its gain — re-validate on
  a panel the tuning never touched (that thread's own +0.065 → +0.023 after
  doing this correctly is the cautionary example). Don't chase differences
  below ~0.005.

## Phase 3 — expand retrieval coverage

**Goal:** CV MRR improves by widening *what* can be retrieved, not yet by
reranking better.

- Attach `samartalwar/casmi-2026-spectral-library-massbankharmonized` as a
  second reference library (broader than Enveda's NP-skewed train set).
- Add mass-shifted analog propagation for Tier-2-style candidates (structural
  relatives via fingerprint similarity + interpretable mass deltas: ±CH2,
  ±OH, ±Hexose, etc.) — this was the single biggest reported improvement in
  the community's shared pipeline.
- **Test curated-NP-database coverage ourselves rather than trusting either
  side.** Two independent top teams directly disagree: one reports curated
  DBs (COCONUT 2.0/ChEBI/LipidMaps) beating blind PubChem expansion, another
  reports those same curated DBs added zero coverage beyond PubChem ∪
  COCONUT and that DB-membership features looked great on their holdout but
  *hurt* on the real leaderboard (their holdout over-represented curated-DB
  molecules — a holdout-construction artifact). Measure our own coverage gap
  directly against our own CV panel before picking a side.
- **Verify:** CV MRR@25 improves over Phase 1's exact-match-only baseline.

## Phase 4 — reranker

**Goal:** CV MRR improves again, this time from ranking order rather than
recall — matches the community's own finding that this is the actual
remaining bottleneck once candidates are retrieved.

- Engineer the feature set the community converged on (~25-30 features):
  spectral similarity scores, analog-propagation scores, precursor mass
  delta, fingerprint similarity, molecular-formula match quality.
- **Prioritize isomer-discriminating features over more mass/formula
  tuning.** RESEARCH.md's "~30 submissions" finding: once a candidate pool
  contains the truth, ~98% of ranking errors are between candidates sharing
  the *exact same formula* — mass-window width is a dead lever on
  timsTOF data specifically (99% of precursor masses already land within
  ~5ppm). The actual signal is connectivity-dependent: in-silico
  fragmentation parsimony (how few bond cleavages a candidate needs to
  explain the observed peaks) — concrete starting recipe from that same
  thread: down-weight 2-bond cleavages (×0.6 vs. 1-bond), linear (not sqrt)
  intensity weighting, 0.005 Da tolerance, H-shifts -2..+3. Hand-written
  bond-breaking chemistry rules (aromatic/benzylic/ring preferences) were
  tested and added nothing — don't spend time hand-crafting those.
- Build a small same-formula "isomer panel" (truth + its best-scoring
  same-formula rivals) as a dev metric alongside overall CV MRR — reported
  as far more sensitive to exactly the errors that matter than aggregate MRR.
- Train both HistGBM and Random Forest rerankers under the same grouped CV
  (by `inchikey14`) and compare — don't assume HistGBM wins just because it's
  more commonly shared; one team reported RF doing better for them.
- **Verify:** CV MRR@25 improves over Phase 3's retrieval-only ranking, and
  any isomer-panel gain must also hold on a held-out panel it wasn't tuned
  against (see Phase 2) before being trusted.

## Phase 5 — Tier 3 (genuinely novel structures)

**Goal:** this is real, open research, not a known recipe — the community
itself flags it as unsolved. Treat it as incremental, not a target to "solve."

- Candidate directions: substructure assembly/scaffold-based generation
  seeded by the FPNet-style fingerprint prediction; or a graceful fallback
  (nearest-known-structure by predicted fingerprint) rather than leaving a
  molecule with zero candidates.
- **Validate only against genuinely-absent molecules, not simulated ones.**
  One team found that deleting a known structure from the database and
  testing whether a generator recovers it overstated real reach by ~3x
  compared to molecules that are truly absent from every database. Hold out
  real gaps for this phase's evaluation, don't simulate them.
- **Verify:** any Tier-3 addition must not regress Tier-1/2 CV performance —
  check per-tier (if we can estimate tier membership locally) before trusting
  an aggregate score improvement.

## Phase 6 — submission hygiene & monitoring

- Watch the leak-remap forum thread; don't let any part of the pipeline
  implicitly depend on current leaderboard behavior.
- Final notebook: single ≤9h run, internet disabled, only our attached
  datasets — test this exact constraint locally before the real deadline.

## What I'd actually do first

Skip the official encoder-decoder baseline entirely — the community's own
experience (slow, frequently-invalid SMILES, nobody competitive uses it) and
our own reading of its code agree it's the wrong starting point. Start at
Phase 0 directly.

## Open decision before writing code

How much of Phase 0-1 should run against the *full* `train.parquet` (2.5M
spectra) vs. a capped sample for faster iteration? The official baseline
itself caps training at 200k spectra on a T4 "so a Kaggle session finishes"
(~22 min vs. ~2.5h for all 1.39M after their own train/val split) — our
retrieval-based approach doesn't train a model in Phase 1 so this matters
less there, but will matter once Phase 4's reranker needs repeated CV runs.

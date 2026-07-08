# Phone Segmentation Research

Research and recommendations for language-agnostic segmentation of isolated
elicitation syllables ("ba", "aba", "ab") into IPA Consonant / Vowel / pause
segments, including rare consonants (clicks, implosives, unusual fricatives),
for building a high-complexity phoneme comparison database.

Compiled July 2026.

## The one-sentence answer

Stop trying to *recognize* the phones and start *aligning* them. Because the
IPA transcription is already known, this is a **constrained forced-alignment**
problem, not segmentation-from-scratch. That single reframe dissolves both the
rare-consonant problem and the "everything inverts" failure mode.

## Documents

| File | Contents |
|------|----------|
| [01-existing-work-audit.md](01-existing-work-audit.md) | What is already in this repo, the seven automatic methods tried, why each failed, what works, data formats, IPA inventory |
| [02-models-and-forced-alignment.md](02-models-and-forced-alignment.md) | The forced-alignment reframe; full model/tool landscape (CLAP-IPA, MFA, torchaudio/MMS, ZIPA, Omnilingual, etc.) with tiers, licenses, install notes |
| [03-segmentation-and-features.md](03-segmentation-and-features.md) | Blind and constrained segmentation (UnsupSeg, DPDP, HMM+SSL); SSL feature extractors and which layer encodes phone identity |
| [04-landmarks-and-transitions.md](04-landmarks-and-transitions.md) | Stevens landmark theory and tools; the centers-vs-boundaries evidence; the direct answer to "should transitions be separate segments?" |
| [05-interactive-learning.md](05-interactive-learning.md) | The "Picasa for phonemes" question: prototype learning over frozen embeddings, iCaRL/FastICARL/FCAC, active learning, continual-learning caveats, build-vs-buy |
| [06-recommended-pipeline.md](06-recommended-pipeline.md) | Consolidated end-to-end pipeline, phased implementation plan, concrete next steps |

## Executive summary

- **Root cause of past failures.** Every method tried (K-means clustering,
  energy/burst heuristics, MFA pattern-scoring) decided *how many* segments
  there are and *which class* each is, using hand-tuned thresholds over
  RMS / ZCR / spectral-centroid features. The known IPA string was thrown away.
  Worse, the MFA attempts used the **English** acoustic model, which has no
  prior for rare consonants, so alignment collapsed toward English-like CV
  (documented in `Montreal/readme.md`: "all but one of the clips labeled final
  are Consonant initial"). Energy heuristics invert because the sign of the
  informative cue **reverses by manner class** (stop burst = abrupt energy rise;
  sonorant junction = gradual change; fricative = high-frequency noise). One
  threshold cannot serve all classes.

- **The reframe.** Supply the known IPA string as a constraint. A forced
  aligner only has to place boundaries between states you have already asserted
  exist, in the order you specified. It never has to *recognize* a click, so
  rare-phone coverage stops being the bottleneck.

- **Top tool to try first: CLAP-IPA / IPA-Aligner** (NAACL 2024, MIT license,
  pip-installable on Windows). The only aligner with a natively IPA token
  vocabulary (clicks, implosives, diacritics, byte-fallback), trained partly on
  DoReCo field-linguistics data.

- **Highest boundary accuracy: MFA 3.x done correctly** (general acoustic model
  + rare-phone remapping, not the English model). This is the VoxAngeles recipe,
  which phone-aligned 95 rare-phone languages this way, then hand-corrected.

- **Most robust custom option: constrained dynamic programming over frozen
  HuBERT features** with the number of segments fixed to the known count.
  Structurally cannot invert, reorder, or hallucinate segments.

- **Transitions: yes, model them separately.** Anchor comparisons on stable
  centers (vowel midpoint, consonant interior, stop-burst instant); represent
  stops/affricates/clicks/implosives as closure + burst (+ transition); keep the
  CV transition as its own labeled zone. A hard C|V boundary is theoretically
  unmotivated and the least reliable point to place.

- **"Live learning" like Picasa face tagging: yes, and it is a small build.**
  Freeze the encoder, represent each functional group as a prototype (mean
  embedding) over confirmed examples, classify new segments by nearest
  prototype, update instantly on each confirmation. This is metric learning /
  nearest-class-mean, not reinforcement learning and not retraining. The
  existing 274 ClipSorter groups are the seed set.

## Recommended pipeline at a glance

1. **Silero VAD** brackets each syllable and finds internal silences/closures
   (the pause class), where energy gates invert.
2. **Constrained alignment of the known sequence** — CLAP-IPA/IPA-Aligner, or
   HuBERT-layer-9 features + forced Viterbi with N fixed. Run both; agreement is
   the confidence signal, disagreement flags the ~20% to hand-tweak.
3. **Landmark refinement for obstruents** — Auto-Landmark (open-source Python)
   to snap to bursts and mark closures; vowel steady-state midpoint as the
   comparison anchor.
4. **Prototype layer** over frozen embeddings that learns the functional groups
   from every confirmation, with active learning choosing what to review next.
5. Write everything to **Praat TextGrids** (via Parselmouth) for hand
   correction in the tool you already use.

See [06-recommended-pipeline.md](06-recommended-pipeline.md) for the detailed plan.

## A note on citations

Anchor sources (Allosaurus, CLAP-IPA, MFA, UnsupSeg, DPDP, HuBERT/WavLM/XEUS,
Stevens landmark theory, iCaRL, prototypical networks, Label Studio) are
well-established and verified. A handful of 2025-2026 arXiv identifiers surfaced
during research are future-dated and were **not** independently verified; these
are flagged inline as "unverified — confirm before relying on it."

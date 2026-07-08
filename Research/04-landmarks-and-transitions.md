# 04 - Landmarks and Transitions

Acoustic landmark detection (Stevens theory) and the direct answer to the
question: should CV boundary/transition regions be defined as separate segments?

## Short answers

- **Detect centers/landmarks, not hard boundaries** - especially for stops,
  affricates, clicks, and implosives. Landmark events (vowel nucleus peak, glide
  minimum, stop closure/burst, voicing on/off) are well-defined and robustly
  detectable; segment *edges* are fuzzy and context-dependent.
- **Yes, model transitions as separate segments.** Anchor comparisons on stable
  centers; represent stops as closure + burst (+ transition); keep the CV
  transition as its own labeled zone. A hard C|V cut is theoretically
  unmotivated and the least reliable point to place.

## Stevens acoustic landmark theory

Kenneth Stevens, JASA 111(4):1872 (2002). Speech is parsed at **landmarks** -
points where the spectrum abruptly changes or reaches an extremum, exploiting
quantal non-linearities in the articulatory-acoustic mapping. Distinctive
features are read off *around* landmarks, not at segment edges. The standard
landmark taxonomy (Stevens/Liu):

- **V** (vowel) - a maximum in low/mid-frequency energy = vowel nucleus. A center.
- **G** (glide) - a minimum in low/mid energy. A center-like extremum.
- **Sc / Sr** - stop closure / stop release (burst).
- **Fc / Fr** - fricative constriction onset / release.
- **Nc / Nr** - nasal closure / release.
- **g** (glottal) - voicing onset / offset.

Landmarks are theorized to be "reliably produced, robustly detectable, and
highly informative", precisely because at these points the acoustics are
*insensitive* to small articulatory variation - the opposite of boundaries,
where the signal changes fastest and is most context-dependent.

## Evidence that centers beat boundaries

- **He, Hasegawa-Johnson et al. 2018** (JASA 143(6):3207; arXiv 1710.09985):
  frames overlapping landmarks carry most of the phone-string information. A
  frame-dropping strategy kept phone error rate within **0.44% of optimal while
  scoring fewer than half (45.8%) of all frames.** Landmark frames are sufficient;
  edges are not the object of interest.
- **Forced-alignment practice treats stop boundaries as arbitrary.** Alignment
  guides instruct annotators to anchor on the visible burst/release and end of
  vowel formant structure, not to pin the fuzzy closure onset; VOT can be < 10 ms
  so tiny edge errors dominate. The closure-burst *transition* is sharp and
  detectable even though the *edges* are not - which is why burst-onset and
  plosion-index detectors target the transition.

## Landmark / VOT detection tools

- **Auto-Landmark** (Zhang et al. 2024, arXiv 2409.07969;
  github.com/Tonyyouyou/Landmark_Dataset) - **the first open-source Python
  landmark toolkit** + a TIMIT-based dataset with precise ground-truth landmark
  timing. Detects 5 typed landmarks (glottal +/-, burst +/-, sonorant +/-,
  voiced-fricative +/-, fricative +/-) with both a signal-processing detector
  (six-band spectrogram, coarse+fine passes) and deep-learning baselines (best
  landmark error rate ~31%). **Use this for obstruent refinement** (replaces the
  dated AutoVOT).
- **Conformer + HuBERT landmark detector** (2026, arXiv 2606.23228, *unverified*;
  demo mateocamara.github.io/acoustic-landmarks) - strongest neural option.
- **AutoVOT** (github.com/mlml/autovot, MLSpeech/AutoVOT) - discriminative VOT
  (burst onset -> voicing onset) from a rough window; ~77% within 2 ms, ~92%
  within 5 ms. Works but is Python-2.7-era (2014) - painful on modern Windows.
  **DeepVOT** (github.com/adiyoss/DeepVOT) is the RNN successor.
- **SpeechMark** (Boyce et al. 2012) and **Liu (1996)** - the foundational
  signal-processing detectors (MATLAB); Auto-Landmark modernizes them in Python.

## Rare-phone specifics (important for this inventory)

Implosives and clicks behave in opposite ways at the burst, and a burst-only
detector fails on one of them:

- **Implosives** (`[ɓ ɗ ʄ]`) may have **no release burst at all** for some
  speakers, with voicing beginning ~58 ms before release (Shimaore data). A
  burst-keyed detector silently fails on exactly those tokens - but the
  **closure-voicing / glottal-onset landmark still fires.** Anchor implosives on
  the voicing-onset landmark, not the burst.
- **Clicks** (`[ǃ ǀ ǂ]`) are the opposite: a loud, salient transient burst that
  is one of the *easiest* landmark targets in the whole signal - but their
  onset/closure edge is genuinely hard to bound. Anchor on the transient center,
  do not fight for the edge. (Dedicated automatic click-landmark literature is
  thin; this is inferred from click acoustics + landmark theory.)

## Should transitions be separate segments? Yes - the precedents

1. **TIMIT itself** annotates every stop as two segments: a **closure**
   (`bcl dcl gcl pcl tcl kcl`) followed by a **release/burst** (`b d g p t k`).
   The gold-standard convention already splits stops rather than forcing one cut.
2. **Diphone synthesis** (Lenzo & Black 2000) cuts units center-to-center
   precisely because "the center of a phonetic realization is the most stable
   region, whereas the transition contains the most interesting phenomena, and is
   the hardest to model."
3. **Articulatory Phonology** (Browman & Goldstein): speech is overlapping
   gestures; a CV transition is a region *shared* by both gestures, not a point.
   A hard boundary there is an artifact.
4. **Modern aligners** (MFA, Kaldi, HTK, charsiu) place a single hard C|V
   boundary, and it is understood to be arbitrary/soft - the formant transition
   is split between C and V by acoustic-model statistics, not a principled event.

## Recommendation for the phone-comparison database

- Store a **stable center window** per phone as the comparison anchor: vowel
  steady-state midpoint; consonant murmur/closure interior; burst instant for
  stops/clicks; voicing-onset for implosives. (This mirrors standard practice of
  measuring vowel formants at the midpoint to avoid transition contamination.)
- Represent stops/affricates/clicks/implosives as **[closure region] + [burst or
  voicing-onset landmark] + [transition-to-V]**, TIMIT-style.
- Keep the **CV transition as its own labeled, hand-correctable zone** rather
  than forcing it into C or V.

This is why the original "center + Hamming window" instinct is correct for
obstruents: detecting a burst *landmark* + a window around it is well-posed,
while detecting a C|V *boundary* is not.

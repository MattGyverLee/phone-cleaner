# 01 - Existing Work Audit

A survey of what is already in the `phone-cleaner` repository: the automatic
segmentation methods attempted, why each failed, what currently works, the data
formats in use, and the IPA inventory involved.

## Goal (reconstructed from the code)

Segment very short recordings (CV, VCV, VC syllables like "ba", "aba", "ab")
into IPA consonant / vowel / pause segments, language-agnostically, for a set of
271 narrowly transcribed rare consonants (implosives, retroflexes, trills,
non-sibilant fricatives, etc.). The vowel is almost always schwa. Source audio
is Glossika clips that first had background music removed by spectral
subtraction. The intended output is a set of comparable, diff-able segments to
seed a machine-learning phoneme database.

## Directory map

| Directory | Purpose | Status |
|-----------|---------|--------|
| `Hamming-Docs` | K-means clustering and Hamming-window analysis | Multiple failed iterations |
| `Montreal` | Montreal Forced Aligner (MFA) attempts | Multiple failed classification iterations |
| `ClipSorter` | Flask web UI for manual clip classification | **Working** (CV/VCV/VC labels) |
| `Sample_Generation` | Synthetic audio (eSpeak IPA, Google TTS) | Exploratory |
| `Music_Removal` | Spectral subtraction to remove background music | **Working** (preprocessing) |
| `Phonetic_Distance` | PHOIBLE parsing, panphon-based phone grouping | Data processing |
| `Auditory_Distance` | Exploratory audio distance analysis | Minimal |
| `Downloaders` | Fetch sample audio | Data collection |

## Methods tried, and why they failed

### 1. K-means clustering (primary attempt)

- Files: `Hamming-Docs/kmeans.py`, `Hamming-Docs/1. Chunkdata.py`
- Features: RMS energy, zero-crossing rate (ZCR), spectral centroid / bandwidth
  / rolloff / flatness, first 5 MFCCs; the advanced version adds F0 (Parselmouth)
  and F1/F2 formants.
- Clustering: StandardScaler + KMeans (k=2 for isolated files, 3-5 otherwise),
  median-filter smoothing, merge segments closer than 50 ms.
- Output: Praat TextGrid, tier "Guessed_Phonemes".
- **Why it failed:** cluster-ID to phoneme-class mapping is manual and per-file
  (`kmeans.py` has placeholder mappings commented "YOU NEED TO DETERMINE THIS
  BASED ON YOUR DATA"). Thresholds are arbitrary (e.g. vowel_score > 2.5). No
  ground truth for rare consonants. Brief high-energy consonants get confused
  with vowel onsets.

### 2. Energy-based peak detection (`simpleHam.py`)

- RMS thresholds (10th/70th percentile), ZCR, spectral centroid; burst detection
  via energy jumps > 2x previous frame; hard-coded rules
  (duration < 80 ms + burst -> stop; duration > 150 ms + low ZCR -> vowel).
- **Why it failed:** hard-coded for the CV "ba" pattern (assumes consonant then
  vowel), weak burst detection, no VCV/VC handling, assumes schwa. Fricatives and
  implosives violate every assumption.

### 3. `ConsonantDetector` (`Hamming-Docs/Hamming-Claude/GroupandHam.py`)

- 3-cluster KMeans (silence / consonant / vowel) over RMS, ZCR, centroid,
  rolloff, 4 MFCCs; sorts clusters by energy; adds +/-50 ms padding and applies a
  Hamming window around detected consonants.
- **Why it failed:** same arbitrary thresholds; assumes a clean 3-cluster
  structure; no acoustic model for rare consonants; no VC/VCV logic.

### 4. Montreal Forced Aligner iterations (`Montreal/`)

Seven scripts of increasing sophistication (`mfa_analysis.py` ->
`improved_pattern_analysis.py` -> `final_accurate_analysis.py`). Approach:
librosa onset detection + energy-drop boundaries, then a weighted "vowel score"
(energy, periodicity, spectral, frequency-band, duration terms), classify vowel
if score > 2.5, then match against expected CV / VCV / VC patterns with heavy
penalties for the wrong segment count.

- **Documented failure** (`Montreal/readme.md`): *"The workflow is good, but the
  classification results are abysmal. All but one of the clips labeled final are
  Consonant initial."* The VC classifier inverts to CV.
- **Root cause:** the MFA runs used the **English** pre-trained acoustic model,
  which has no acoustic model for `[ɗ]`, `[ʙ]`, `[χ]`, `[ʂ]`, etc. Forced
  alignment degrades and biases toward English-like structure. This is a
  configuration error (wrong acoustic prior), not a flaw in forced alignment
  itself. Additional issues: Unicode/UTF-8 filename handling; only one pattern
  tested per file instead of all three.

### 5-7. Supporting variants

Further K-means and energy variants share the same magic-number thresholds and
the same lack of a rare-phone acoustic model.

## What works

- **ClipSorter** (`ClipSorter/clipsort.py`): a Flask app that groups clips by
  phoneme (regex over filename) and lets the user label each `initial` (CV),
  `medial` (VCV), `final` (VC), or `other`, saving to
  `classification_progress.json`. **274 phoneme groups have been manually
  classified.** This is a genuine asset: it can serve as ground truth / a
  validation set / the seed for prototype learning (see
  [05-interactive-learning.md](05-interactive-learning.md)).
- **Music_Removal** (`Music_Removal/MusicMatchRemoval.py`): librosa +
  spectral subtraction. Solid preprocessing; keep it.

## Feature extraction used across all attempts

RMS energy, ZCR, spectral centroid / rolloff / flatness, MFCCs (4-13), F0
(Parselmouth), F1/F2 (Praat Burg). Consistent gaps:

- No **voice onset time (VOT)** or explicit burst/landmark features.
- No **spectral envelope / spectral peaks** for specific rare consonants.
- Formant extraction returns NaN for unvoiced/rare phones and is replaced by 0
  (invalid); F0 is unreliable for brief unvoiced consonants.
- All thresholds are hand-tuned magic numbers, learned from nothing.

## Data formats in use

- **Praat TextGrid** (from the K-means / Chunkdata pipeline): tier
  "Guessed_Phonemes", intervals `(start, end, label)`.
- **Python dict / JSON** (Montreal scripts, ClipSorter): `{start, end, phoneme,
  type, confidence, features}` in seconds.
- **Classification JSON** (ClipSorter `classification_progress.json`):
  `{"completed_groups": ["[ b ] voiced...", ...]}`.

## IPA inventory

271 narrowly transcribed consonants sourced from PHOIBLE. Examples:
`[b]` voiced unaspirated bilabial stop; `[pʰ]` unvoiced aspirated bilabial stop;
`[ɗ]` voiced apical alveolar implosive; `[ɲ]` palatal nasal; `[ɱ]` labiodental
nasal; `[ɽ]` subapical retroflex; `[ʙ]` bilabial trill; `[ɸ]`/`[β]` bilabial
non-sibilant fricatives. The vowel is schwa in the CV/VCV/VC frames.

The challenge these pose: no ASR/aligner training corpus contains meaningful
acoustics for clicks, implosives, or ejectives, so any method that must
*recognize* them will fail. See [02-models-and-forced-alignment.md](02-models-and-forced-alignment.md)
for why forced alignment sidesteps this.

## Takeaways that drive the recommendation

1. The known IPA string is being discarded; it should be the primary constraint.
2. The MFA architecture was right; the English acoustic model was wrong.
3. Boundary placement for stops/rare consonants is inherently fuzzy and should be
   probabilistic / landmark-based, not a single hard cut.
4. The 274 manually-classified groups are training/validation signal, not
   throwaway labels.

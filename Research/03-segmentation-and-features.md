# 03 - Segmentation Methods and SSL Features

Blind and constrained phoneme boundary detection, and the self-supervised
speech features that make them work. This is the "build it yourself" path and
the cross-check for the forced aligners in
[02-models-and-forced-alignment.md](02-models-and-forced-alignment.md).

## Why energy/spectral heuristics invert

The sign of the informative cue **reverses by manner class**:

- a **stop** landmark is an abrupt energy *increase* (burst after near-silence);
- a **voiced-stop closure** is an energy *minimum* that looks like a pause;
- a **sonorant/glide** junction is a *gradual* change with no discontinuity;
- a **fricative** cue is high-frequency noise onset, not low-band energy.

A single energy/spectral-change threshold cannot serve all classes at once - tune
it for stop bursts and it fires twice on transients and misses glide junctions;
tune it for sonority and it inverts on obstruents. This is a category error, not
a tuning problem. The fixes: (1) use SSL features whose frame-to-frame distance
tracks *phonetic* change, and (2) use the known segment count/order as a
constraint.

Resolution note: HuBERT/wav2vec2 features have a 20 ms frame stride, so any
SSL-feature method has ~20 ms intrinsic boundary resolution - which is the
standard evaluation tolerance and fine for a hand-correctable first pass.

## Constrained segmentation when the sequence is known (the key idea)

You are not *detecting* boundaries; you are *aligning a known short label
sequence* to audio. That is a solved, robust operation and needs no
language-specific training:

1. Extract frozen SSL features (HuBERT layer ~9).
2. Run Viterbi / DP over a strictly left-to-right state graph whose states are
   your known sequence (e.g. `pause -> C -> V -> pause`), each state a
   mandatory, self-looping segment.
3. Add a per-class duration prior to prevent 1-frame segments.
4. Assign the known IPA labels to the resulting segments left-to-right.

Because the path is forced through exactly your states in order, the decoder
**cannot invert, reorder, or hallucinate segments** - it only places the
transition frames. And it never has to *recognize* a click; it only places a
boundary between two states you asserted exist.

Published realizations:

- **Simple HMM with Self-Supervised Representations** (2024, arXiv 2409.09646) -
  k-means over HuBERT layer-9 features + Viterbi HMM with a duration penalty
  (HMM-DP) **or a fixed segment count (HMM-Nseg)**. The HMM-Nseg mode is exactly
  "I know how many segments there are". TIMIT F1 82.1 / R 84.4; Buckeye F1 78.1.
  Simple enough to reimplement in ~a day; no official repo located.
- **torchaudio `forced_align`** - a clean CTC/Viterbi forced-alignment primitive,
  actively maintained. Feed it your own SSL-feature emission scores over
  C/V/pause classes (not the romanized MMS token set).
- **DPDP** (Kamper 2022-23, github.com/kamperh/vqwordseg) - duration-penalized DP
  over SSL/VQ features with an option for a **prespecified number of segments**.
  Weak as an open *phone* segmenter (Buckeye phone F1 ~36) but the fixed-N DP
  skeleton is reusable.

## Blind (unsupervised) boundary detectors - for cross-checking

Useful to run alongside a forced aligner: detect boundaries with no labels, then
keep the N-1 most prominent peaks for the known N segments. All F1 at 20 ms
tolerance on continuous TIMIT/Buckeye; isolated syllables should meet or beat
these.

| Method | Year | TIMIT F1 | Code | Notes |
|--------|------|----------|------|-------|
| UnsupSeg (Kreuk) | 2020 (arXiv 2007.13465) | 83.7 | github.com/felixkreuk/UnsupSeg (MIT) | CNN + noise-contrastive; peak-picking on a frame-dissimilarity curve. Lowest-friction "find boundaries" tool. Run on HuBERT features and keep the top N-1 peaks. |
| SCPC (Bhati) | 2021 (arXiv 2106.02170) | 85.3 | research code | Hierarchical CPC with a differentiable segmenter. Notes vowel-vowel junctions are hardest. |
| Strgar & Harwath | 2022 (arXiv 2211.01461) | ~82 (strict) | github.com/lstrgar/self-supervised-phone-segmentation (GPL-3) | Best-maintained; ships wav2vec2 + HuBERT checkpoints; fairseq dependency (use WSL). |
| SegFeat (supervised) | 2020 | SOTA (needs labels) | github.com/felixkreuk/SegFeat (MIT) | Can consume the phone transcription; pinned to torch 1.4 (old). |
| DPDP / vqwordseg | 2022-23 | (phone ~36) | github.com/kamperh/vqwordseg (MIT) | Unit-discovery tool; reuse only the duration-penalized DP idea. |

## Self-supervised feature extractors: which layer encodes phones

Do **not** use the final layer for phone work. Phone identity lives mid-stack.

| Model | HF id | Phone layer | Frame rate | License | Notes |
|-------|-------|-------------|-----------|---------|-------|
| HuBERT base | `facebook/hubert-base-ls960` | ~7-9 | 20 ms | permissive | Easiest phone-quality features in pure `transformers` (no fairseq). Community default. |
| HuBERT large | `facebook/hubert-large-ll60k` | mid-upper | 20 ms | permissive | |
| wav2vec2 / XLS-R | `facebook/wav2vec2-xls-r-300m` | base ~6; large/XLS-R ~15-19 | 20 ms | Apache-2.0 | XLS-R covers 128 languages - best wav2vec2-family multilingual choice. |
| WavLM | `microsoft/wavlm-large` | upper-mid (~18-24) | 20 ms | MIT | Denoising pretraining -> most robust to noisy field recordings. |
| XEUS (CMU) | `espnet/xeus` | pick empirically | 20 ms | CC-BY-NC-SA (non-commercial) | 4057-language coverage - most likely to generalize to clicks/implosives. Feature extractor only; runs through a forked ESPnet (fiddly on Windows; prefer WSL). 577M params -> GPU. |
| w2v-BERT 2.0 | `facebook/w2v-bert-2.0` | mid-stack (validate) | ~20 ms | MIT | 143+ languages; first-class in `transformers` - easiest large multilingual model to run on Windows. |

Layer-wise evidence: Pasad et al. (arXiv 2211.03929, 2107.04734) show
discrete-target models (HuBERT, WavLM) keep phone information high in the stack,
while wav2vec2 peaks mid-stack then decays. HuBERT layer 7 maximizes phone
purity (arXiv 2106.07447). **Pick the layer empirically on your own data**: extract
each layer, run the segmenter, measure agreement against a few hand-corrected
files, keep the winner. Published "best layer" numbers shift +/-2 by probe method.

## Practical VAD and posteriorgram tooling

- **Silero VAD** (github.com/snakers4/silero-vad) - lightweight, CPU-only,
  pip-installable, actively maintained. Best drop-in for speech-vs-silence: it
  brackets each syllable and finds internal pauses/closures far better than an
  energy gate. Use it for the **pause class** and outer bracketing.
- **SpeechBrain VAD** (`speechbrain/vad-crdnn-libriparty`) - heavier alternative
  with probability contours.
- **Allosaurus** (github.com/xinjli/allosaurus) - universal ~230-phone IPA
  recognizer; outputs top-K per-frame candidates (a sparse posteriorgram) and
  approximate timestamps; inventory can be restricted to the known phones. Use as
  a **second opinion** and for posteriorgram-DTW template matching against a
  recorded reference exemplar per phone. Not a primary boundary source (coarse).
- **Parselmouth** (Python Praat) - use for **measuring and writing TextGrids**,
  not for finding boundaries (its intensity/formant primitives are the ones that
  invert).

## Recommendation

Constrained left-to-right alignment over frozen HuBERT-layer-9 features
(HMM-Nseg, or torchaudio `forced_align` with your own emissions), cross-checked
against UnsupSeg-style peak detection (keep the top N-1 peaks). Use Silero VAD
for the pause class. This needs no training, exploits the known sequence, and is
robust to rare phones because it aligns rather than recognizes.

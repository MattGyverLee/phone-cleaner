# 06 - Recommended Pipeline

The consolidated end-to-end approach, a phased implementation plan, and concrete
next steps. Draws together the reframe
([02](02-models-and-forced-alignment.md)), constrained alignment over SSL
features ([03](03-segmentation-and-features.md)), landmark refinement
([04](04-landmarks-and-transitions.md)), and prototype learning
([05](05-interactive-learning.md)).

## Design principles

1. **Align, do not recognize.** Supply the known IPA string as a hard constraint.
2. **Constrain by the known count/order.** Force a left-to-right path through the
   known C/V/pause states so the segmenter cannot invert or hallucinate.
3. **Freeze the encoder.** Use SSL features as a fixed representation; never
   backprop into it (avoids forgetting; makes learning instant).
4. **Detect centers/landmarks, not hard edges** for obstruents and rare phones;
   keep transitions as their own zones.
5. **Learn from every confirmation** via prototypes; **route review** via active
   learning; **display confidence**, not hard labels.
6. **Two independent aligners agreeing = confidence.** Disagreement flags the
   ~20% to hand-correct.

## The pipeline

```
audio (music already removed by Music_Removal)
  |
  v
[1] Silero VAD ................ bracket the syllable; find pause/closure regions
  |
  v
[2] Constrained alignment .... align the KNOWN IPA sequence to audio
     |  primary:   CLAP-IPA / IPA-Aligner  (IPA-native, MIT, pip)
     |  or/and:    HuBERT layer-9 features + forced Viterbi, N fixed (HMM-Nseg)
     |  cross-check: UnsupSeg-style peak detection, keep top N-1 peaks
  |
  v
[3] Landmark refinement ...... Auto-Landmark: snap obstruents to burst/closure;
     |                          implosives -> voicing-onset landmark;
     |                          vowels -> steady-state midpoint (center window)
  |
  v
[4] Prototype layer .......... frozen-embedding NCM/prototype classifier over the
     |                          274 ClipSorter groups; rank candidates by nearest
     |                          prototype; update instantly on each confirmation;
     |                          active learning picks what to review next
  |
  v
[5] Output ................... Praat TextGrids (via Parselmouth):
                               tiers for C / V / pause + transition zones +
                               center windows; hand-correct in Praat/ELAN
  |
  v
(corrections feed back: Loop A -> prototypes instantly; Loop B -> mfa adapt, batch)
```

## Phased implementation plan

### Phase 0 - validation harness (do this first)

- Use the 274 ClipSorter groups as ground truth. Build a small script that scores
  any segmenter's output against a hand-corrected subset (boundary error in ms,
  class accuracy). Every later choice (which model, which HuBERT layer) is decided
  by this harness, not by intuition.

### Phase 1 - forced-alignment first pass

- Install CLAP-IPA (`git clone github.com/lingjzhu/clap-ipa && pip install .`).
- Run `anyspeech/ipa-align-base-phone` on the existing `iso_[b]` / `iso_[f]`
  samples in `Hamming-Docs`; emit TextGrids from the notebook.
- Measure against the validation harness. Note behavior on single-phone isolated
  clips (DTW degenerate case) vs. "aba"/"ab".

### Phase 2 - constrained DP cross-check

- Extract frozen HuBERT layer-9 features (`transformers` + `torchaudio`, pure
  Windows Python).
- Implement left-to-right Viterbi forced through the known state sequence with a
  per-class duration prior (HMM-Nseg style), or drive `torchaudio.functional.forced_align`
  with your own C/V/pause emissions.
- Compare against CLAP-IPA; treat agreement as confidence, disagreement as the
  review queue.
- Empirically pick the HuBERT layer using the validation harness.

### Phase 3 - landmark refinement

- Add Auto-Landmark (github.com/Tonyyouyou/Landmark_Dataset) for obstruents:
  snap to burst/closure landmarks, mark implosive voicing-onset, take vowel
  steady-state midpoints as center windows.
- Emit center windows + transition zones as separate TextGrid tiers.

### Phase 4 - prototype learning layer

- `embed.py` (frozen HuBERT), `prototypes.py` (NCM + herding + FAISS + prototype
  JSON). Seed prototypes from the 274 groups.
- Rank unlabeled clips by nearest prototype; show label + distance as a
  hypothesis. Each confirmation appends an exemplar and recomputes the prototype
  (O(1)).
- Guardrails: cosine + feature whitening; cap exemplars per group; per-group
  thresholds; confidence display.

### Phase 5 - active learning + boundary adaptation

- Add uncertainty sampling (MC-dropout or committee) to order the review queue.
- Optionally add MFA 3.x with a general acoustic model + rare-phone remapping,
  and use `mfa adapt` on accumulated corrected TextGrids for boundary precision
  (Loop B, batch every ~50-100 corrections).

### Phase 6 - UI decision

- Either extend the existing ClipSorter Flask app with the prototype backend and
  a ranked-candidates view (recommended, lowest risk), or host the identical
  prototype logic behind a Label Studio ML backend for a polished multi-user
  waveform UI.

## Effort estimate

- Phases 1-2 (forced-alignment first pass + constrained-DP cross-check): a few
  days; pure Windows Python (no fairseq/Kaldi needed for HuBERT + CLAP-IPA +
  torchaudio `forced_align`).
- Phases 3-4 (landmarks + prototypes): another few days; the prototype layer is
  ~a few hundred lines.
- Phase 5-6 (active learning, MFA adaptation, UI): incremental.

## Windows install notes

- Clean pip/torch installs: HuBERT/`transformers`, CLAP-IPA, torchaudio
  `forced_align`, Silero VAD, Allophant, MultIPA, Auto-Landmark, scikit-learn,
  river, FAISS.
- Conda recommended: MFA (bundles Kaldi).
- Prefer WSL2/Linux: ZIPA *training* (k2/icefall/kaldifeat), XEUS (forked
  ESPnet), Strgar & Harwath (fairseq). Inference-only paths mostly avoid these.
- AutoVOT is Python-2.7-era; use Auto-Landmark instead.

## Immediate next step

Scaffold the Phase 1 + Phase 4 seed as an MVP: a script that loads
`ClipSorter/classification_progress.json`, embeds the confirmed clips through
frozen HuBERT, builds the 274 prototypes, and ranks a folder of unlabeled clips
by nearest prototype - so the "find this group in the crowd" behavior is visible
on real data before wiring it into a UI. In parallel, run CLAP-IPA on one
`iso_[b]`/`iso_[f]` sample and emit a TextGrid to see the first-pass alignment.

## Key links

- CLAP-IPA: github.com/lingjzhu/clap-ipa - arXiv 2311.08323
- VoxAngeles (precedent): arXiv 2403.19509 - github.com/pacscilab/voxangeles
- MFA: montreal-forced-aligner.readthedocs.io
- HMM + SSL constrained segmentation: arXiv 2409.09646
- UnsupSeg: github.com/felixkreuk/UnsupSeg - arXiv 2007.13465
- HuBERT: huggingface.co/facebook/hubert-base-ls960 - arXiv 2106.07447
- Auto-Landmark: github.com/Tonyyouyou/Landmark_Dataset - arXiv 2409.07969
- iCaRL: arXiv 1611.07725 - FastICARL: arXiv 2106.07268 - FCAC: github.com/vinceasvp/FCAC
- Silero VAD: github.com/snakers4/silero-vad
- Allosaurus (bootstrap): github.com/xinjli/allosaurus
- Label Studio ML backend: labelstud.io/guide/ml_create

# 02 - Models and Forced Alignment

The core reframe, and the full landscape of models and tools evaluated for
"known IPA string -> phone boundaries" on isolated syllables with rare
consonants.

## The reframe: forced alignment, not recognition

Because the IPA transcription is already known, the problem is **forced
alignment** (align a given phone sequence to audio), not **open recognition**
(guess what phones are present).

- Open recognizers (Allosaurus, ZIPA, POWSM) try to identify phones and skew
  toward the phonology of their training languages. That is what produced the
  skew/inversion in past attempts.
- Forced alignment supplies the labels; the model only places boundaries. It
  never has to *recognize* a click, so rare-phone coverage stops being the
  bottleneck. Forced alignment also degrades gracefully: even a weakly-grounded
  model usually places a boundary roughly correctly for a burst/transient.

This is also why the past MFA attempts failed: the architecture (forced
alignment) was correct, but they used the **English** acoustic model, which has
no prior for rare consonants.

### Connection to the CharScan project

The same principle drives the sibling OCR project (identify characters by their
*form*, not their *environment*): suppress the language-model/context prior. In
OCR a next-letter language model corrupts an unfamiliar glyph; in phone work a
language-specific acoustic/phonotactic model corrupts a rare consonant. In both,
the winning move is a strong explicit per-item prior (the known glyph set / the
known IPA string) feeding a model used as a pure feature extractor, with no
autoregressive context leaking bias in.

## The rare-phone reality check

- **No public model has meaningful acoustic training on clicks, implosives, or
  ejectives.** These appear in a handful of languages thin-to-absent in
  Common Voice, FLEURS, and G2P-derived corpora.
- **eSpeak-NG-based models cannot even represent them.** The
  `facebook/wav2vec2-lv-60-espeak-cv-ft` vocabulary (~391 tokens) has no clicks,
  implosives, or ejectives. Same limitation for any eSpeak-target model.
- **CTC-romanization aligners collapse them.** torchaudio `MMS_FA` and
  `ctc-forced-aligner` romanize IPA through `uroman` to `a-z` + apostrophe;
  clicks/implosives/diacritics are dropped or merged before alignment.
- **Only CLAP-IPA and Allosaurus carry rare IPA as first-class tokens.** MFA
  carries them only if you map them to attested phones. This is decisive.

## Tier 1 - purpose-built forced aligners (best fit)

### CLAP-IPA / IPA-Aligner  (start here)

- "The taste of IPA", NAACL 2024. arXiv 2311.08323. MIT license.
- Repo: github.com/lingjzhu/clap-ipa. Models: `anyspeech/clap-ipa-{tiny,base,small}-phone`,
  `anyspeech/ipa-align-{tiny,base,small}-phone`. Example:
  `forced_alignment_example.ipynb`.
- Dual encoders (phone + speech) with contrastive training; IPA-Aligner is
  fine-tuned with a Forward-Sum loss as a neural forced aligner. Alignment comes
  from a speech/phone similarity matrix + DTW.
- **IPA-native:** ~450-token SentencePiece vocabulary covering all base IPA
  symbols, diacritics, tones, tie bars, with byte-fallback for unseen symbols.
  `phoneset.txt` includes clicks, all five implosives, unusual fricatives.
- Trained on IPAPack (115+ languages) incl. FLEURS-IPA, MSWC-IPA, and
  **DoReCo-IPA** (endangered-language field data). Validated zero-shot on unseen
  DoReCo languages.
- Gives phone-level AND word-level boundaries; ~10-20 ms resolution.
- Windows-friendly: `git clone ... && pip install .`, pure PyTorch/HF.
- **Caveats:** phone-boundary precision is only fair (TIMIT phone-boundary
  F1 ~= 61 at 20 ms; much stronger at word boundaries), which matches an
  ~80%-and-hand-correct goal. The DTW wants a *sequence*, so "aba"/"ab" are ideal
  but a single isolated phone is a degenerate case to test carefully. Model cards
  are sparse; work from the notebook.

### VoxAngeles / UCLA Phonetics Lab Archive segmentation  (the precedent to read)

- LREC-COLING 2024. arXiv 2403.19509. Corpus: github.com/pacscilab/voxangeles
  (CC-BY-NC 4.0). ACL 2024.lrec-main.1114.
- Phone-level forced alignment of the UCLA archive (95+ languages of isolated
  words/segments with IPA transcriptions) using **MFA + a general acoustic
  model**, then **hand-audited**. This is exactly the "80% first pass,
  hand-correct the rest" workflow, on real rare-phone data across the IPA chart.
- Read this first; it tells you what error rates to expect and is the best
  available fine-tuning/eval set for rare-phone alignment.

## Tier 2 - MFA done correctly (highest boundary accuracy)

### Montreal Forced Aligner 3.x

- Docs: montreal-forced-aligner.readthedocs.io. Kaldi HMM-GMM. MIT.
- Best boundaries in the field (mean boundary error < 15 ms on clean data).
  Handles short single utterances well.
- MFA 3.0 uses a cross-linguistically harmonized narrow-IPA phone set and has
  **phone-set remapping** (many-to-one) so rare phones can map to the nearest
  attested phone. It also has an `mfa adapt` workflow: align, inspect per-utterance
  quality metrics, correct in Praat, adapt the acoustic model on corrections,
  re-run (see [05-interactive-learning.md](05-interactive-learning.md)).
- Install via conda on Windows (`conda-forge montreal-forced-aligner`) - smoothest
  path since it bundles Kaldi.
- **What you must supply:** a per-syllable pronunciation dictionary, a general
  acoustic model (e.g. Global English / multilingual IPA), and an IPA ->
  model-phone remapping for rare consonants (accepting that click acoustics are
  approximated). This is the VoxAngeles recipe.

## Tier 3 - open recognizers with a CTC head (align via torchaudio)

General recipe: run the model to get frame-level CTC emissions (log-probs over
the phone vocab), then `torchaudio.functional.forced_align(emission, tokens)`
with `tokens` = your IPA string mapped to the model vocab, then
`merge_tokens()` for per-phone frame spans. ~20 ms resolution.

| Model | Year | IPA? | Timestamps | Notes |
|-------|------|------|-----------|-------|
| ZIPA / IPAPack++ | ACL 2025 (arXiv 2505.23170) | Yes | via CTC | MIT; best PFER; but trained on G2P (citation-form) transcriptions and uses subword unigram tokens (boundaries land on subwords). Use `crctc` checkpoints. Training needs k2/icefall (painful on Windows); inference is fine. |
| PhoneticXEUS | Interspeech 2026 (arXiv 2603.29042, *unverified*) | Yes | exposes CTC logits | Self-conditioned CTC on XEUS; newest/most accurate open recognizer. Verify license. |
| Allophant | Interspeech 2023 | Yes | via CTC | `pip install allophant`, Apache-2.0. Can decode a **custom user-supplied phone inventory zero-shot** via articulatory features (Allophoible) - good for constraining output to the known inventory. |
| Wav2Vec2Phoneme (espeak) | 2021 (arXiv 2109.11680) | eSpeak | via CTC | `facebook/wav2vec2-lv-60-espeak-cv-ft`. **Cannot represent clicks/implosives** - hard blocker for rare phones. |
| MultIPA | Interspeech 2023 | Yes (panphon) | via CTC | github.com/ctaguchi/multipa. Only 7 training languages, ~9 h; superseded by ZIPA. |
| Allosaurus | 2020 (arXiv 2002.11800) | ~230-phone IPA | approximate | github.com/xinjli/allosaurus. Recognizer, not aligner, but fine-tunes from ~10 examples and can restrict its inventory; best **bootstrap pre-labeler** for rare phones. Verify click coverage with `list_phone`. |

## Avoid / not applicable

- **torchaudio `MMS_FA` and `ctc-forced-aligner`** - romanize IPA to `a-z`;
  rare consonants collapse. Useful only for the `forced_align` API itself
  (reuse it with your own IPA-capable emissions). MMS weights are CC-BY-NC.
- **charsiu** - released models are English + Mandarin only; IPA support is
  roadmap. The frame-classification alignment architecture is right, but you
  would need to retrain for your inventory. github.com/lingjzhu/charsiu, MIT.
- **WhisperX** - grapheme/character alignment, English-centric, weakest phone
  precision; also assumes you do not have the transcript.
- **Meta Omnilingual ASR** (Nov 2025, arXiv 2511.09690, Apache-2.0) - orthographic
  transcription over 1600+ languages, **no timestamps, no IPA, no alignment**.
  Not usable for boundaries.
- **"SAM Audio" / Segment Anything for audio** (Dec 2025) - sound-source
  separation, not phonetics. Irrelevant.
- **POWSM** (arXiv 2510.24992), **WhisperIPA/neurlang**, **CrisperWhisper** -
  seq2seq/decoder models; recognize IPA or words but give no reliable
  frame-level phone timestamps.
- **P2FA / pyfoal / Gentle** - English-only. **Google USM** - orthographic,
  closed, no alignment.

## License quick-map

- MIT: CLAP-IPA, charsiu, ZIPA, UnsupSeg (permissive).
- Apache-2.0: wav2vec2-espeak, Allophant, Omnilingual ASR, MultIPA.
- CC-BY-NC (non-commercial): XEUS, MMS/MMS_FA weights, VoxAngeles data.
- MFA: MIT (code); acoustic models vary.

## Ranked recommendation

1. **CLAP-IPA / IPA-Aligner** - only IPA-native, phone-level, transcript-aligning,
   any-language tool; MIT; Windows-OK. Prototype first.
2. **MFA 3.x** with a general acoustic model + rare-phone remapping - best
   boundaries; the VoxAngeles recipe.
3. **Constrained DP over frozen SSL features with N fixed** - see
   [03-segmentation-and-features.md](03-segmentation-and-features.md); most
   robust to rare phones because it never recognizes, only places transitions.

Run two of these and treat agreement as the confidence signal.

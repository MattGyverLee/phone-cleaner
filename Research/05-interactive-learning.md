# 05 - Interactive / Incremental Learning ("Picasa for phonemes")

The question: is there a "live" model that learns better alignment/classification
from progressive human feedback (confirming functional groups), instead of
one-shot batch processing corrected at the end - like Picasa/Google Photos face
tagging that gets better at finding a person as you verify more examples?

Answer: **yes, it is established practice, it is a small build, and it is not
reinforcement learning and not retraining.**

## What Picasa actually did (the mechanism to copy)

Picasa face tagging is a **nearest-mean-of-exemplars / prototype classifier over
a frozen embedding space**:

1. A **frozen** deep network turns each face into an embedding (a fixed vector).
   This never changes as you tag.
2. A **cheap, instantly-updatable layer** on top - the mean embedding of the
   faces you confirmed as a given person (a *prototype*). Confirming one more
   face nudges that mean. Classify a new face by nearest prototype.

The heavy representation stays frozen (no expensive training, no catastrophic
forgetting); the thing that "learns your categories" is a lightweight prototype
layer that updates in milliseconds per confirmation. This is **metric learning /
prototype learning**; the incremental version is **nearest-class-mean (NCM)** or
**prototypical networks**.

Mapping to this project:

| Picasa | This project |
|--------|--------------|
| Face image | Audio segment (or frame) |
| Frozen face-embedding net | Frozen SSL encoder (HuBERT layer 9 / XEUS / CLAP-IPA) |
| Person identity | A functional group / narrow-IPA category |
| Confirmed faces | The 274 ClipSorter groups - the seed set already exists |
| "Find them in a crowd" | Find/align that phone in new recordings |
| Confirming one more face | Correcting/confirming one more segment nudges its prototype |

You do **not** want reinforcement learning here (reward-driven, data-hungry,
unstable) or repeated fine-tuning. You want prototype bookkeeping over a frozen
encoder.

## Two feedback loops (do not conflate them)

- **Loop A - feedback on WHAT (classification):** "this segment is `[ɗ]`, not
  `[d]`." Pure Picasa loop; genuinely live. Prototype-over-frozen-embeddings
  updates instantly, needs a handful of examples per class, improves recognition
  of that rare consonant with each confirmation. Easy, high payoff.
- **Loop B - feedback on WHERE (boundary placement):** "the burst is 15 ms
  later." Improves by periodic batch fine-tuning of a small alignment adapter, or
  via MFA's `mfa adapt` on corrected TextGrids. A slower (batch) loop, not
  per-click.
- **Loop A largely drives Loop B for free.** In the constrained-DP aligner,
  boundaries are derived from frame-level class scores; as the prototype layer
  gets better at telling `[ɗ]`-frames from vowel-frames, the boundaries the DP
  places sharpen automatically. Fall back to `mfa adapt` only for the last bit of
  boundary precision.

## The algorithm is settled

| Method | Year / link | Role |
|--------|-------------|------|
| **iCaRL** | CVPR 2017, arXiv 1611.07725 | The formal Picasa mechanism: nearest-mean-of-exemplars + herding (keep the few exemplars whose mean best approximates the class) + optional distillation. Keep the encoder frozen and you skip distillation -> ~zero representational forgetting. |
| **FastICARL** | Interspeech 2021, arXiv 2106.07268 | iCaRL adapted for **audio**, on-device, kNN over quantized exemplars. Confirms the pattern works for incrementally-added audio classes. |
| **FCAC** (few-shot class-incremental audio) | IEEE TMM 2023, arXiv 2305.19539; code github.com/vinceasvp/FCAC | Closest published match: frozen embedding + an **expandable prototype classifier**; add a new class from a few labeled examples, no backbone retraining, no forgetting. Reference code to lift from (also DPL-FCAC: github.com/chester-w-xie/DPL_FCAC). |
| **Prototypical Networks** | NeurIPS 2017, arXiv 1703.05175; KWS: arXiv 2007.14463 (code github.com/ArchitParnami/Few-Shot-KWS) | Per-class mean embedding, classify by nearest prototype. `sicara/easy-few-shot-learning` implements NCM + prototypical nets. |
| **NCM + replay** | arXiv 2103.13885 | Simplest viable core; NCM classifier reduces recency bias; small replay buffer keeps old classes calibrated. |
| **SGDClassifier.partial_fit / river** | scikit-learn; riverml.xyz | The practical "instant update per confirmed example" linear layer over frozen embeddings. Boring, mature, Windows-trivial. |

Speech-domain precedent that the whole approach is established: Rainbow Keywords
(incremental KWS, arXiv 2203.16361), AnalyticKWS (exemplar-free class-incremental
KWS, 2025), Samsung online continual KWS. "Keyword" ~= "phone functional group"
mechanically.

## Active learning - get better *faster*

Have the tool surface the segments it is least confident about (near a
prototype boundary) so review time goes to informative cases - which tend to be
the rare consonants.

- **Epistemic-uncertainty sampling** (MC-dropout; arXiv 2306.02105) - ~27% error
  reduction with ~45% less labeled data on ASR. After each correction, recompute
  uncertainty and select the next batch.
- **Query-by-committee** - a lighter ensemble-disagreement alternative.

## Catastrophic-forgetting caveat (specific to rare phones)

The dominant failure mode when adding rare categories incrementally is
**imbalanced forgetting**: a trained classifier drifts toward frequent classes
and drowns the 3-example clicks. Two design choices neutralize it:

1. **Keep the encoder frozen.** No backprop into HuBERT -> ~zero representational
   forgetting; the dominant failure mode disappears. This is the strongest reason
   to use prototypes rather than "retrain a head each time a group is added"
   (that retraining *is* the trap).
2. **Make rare classes compete fairly:** cosine distance with feature
   whitening / mean-normalization so a 3-exemplar click ranks against a
   300-exemplar vowel; cap exemplars per group (herding); per-group distance
   thresholds; show the linguist **confidence/distance**, not a hard label, so
   early rare-phone guesses read as hypotheses.

## Build vs. buy

No off-the-shelf tool delivers "Picasa-for-phones" end to end, but the algorithm
is settled and training-free at add-time, so the build is small.

- **Recommended: extend ClipSorter.** Add `embed.py` (frozen HuBERT via
  `transformers` + `torchaudio`), `prototypes.py` (NCM + herding + a FAISS index +
  a JSON of prototypes), and a "candidates ranked by nearest prototype" view. The
  274 existing confirmations initialize the prototypes in one pass. Adding a rare
  group = drop in 3-10 confirmed vectors and recompute the mean (O(1)). Lowest
  risk, fully under your control, and it turns confirmations into a live training
  signal.
- **Alternative for a polished multi-user UI: Label Studio ML backend.** Put the
  identical prototype logic behind `LabelStudioMLBase`: `predict()` returns
  pre-labeled audio regions with timestamps, `fit()` recomputes prototypes on each
  confirmation. Inherit its audio-regions template and active-learning task
  ordering. Docker recommended on Windows. `labelstud.io/guide/ml_create`.
- **Prodigy** (explosion.ai) - paid equivalent with slicker active-learning
  ergonomics; custom audio recipe with an `update` callback.
- **Doccano** (text-only), **Audino / ELAN / Praat** (no learn-from-feedback
  backend) - ELAN is your eventual interchange target, not the learning engine.
- **Bootstrap rare phones with Allosaurus** - fine-tunes from ~10 examples and can
  *propose* candidate segments/IPA labels for confirmation, filling the prototype
  store faster for never-seen sounds.

## What to avoid

Do not build this as "fine-tune / retrain a classifier head every time a group
is added." That is the imbalanced-forgetting trap for rare phones, needs a
training loop, and buys nothing over prototypes when the encoder is frozen. Keep
the encoder frozen; keep the "learning" as prototype/exemplar bookkeeping.

# CUS Reader 0.8.0

Correctness repairs to image decoding, honest interval statistics on the
promotion gate, a learner with enough capacity to be worth measuring, and a
label schema the supplied dataset can actually populate.

**This release does not establish diagnostic accuracy.** No real case was
decoded while preparing it. Every claim below is about software behaviour.
The clinical work still outstanding is unchanged and is listed in
[VALIDATION_PROTOCOL.md](VALIDATION_PROTOCOL.md) and the "Work needed to
establish accuracy" section of
[ACCURACY_AUDIT_2026-09-30.md](ACCURACY_AUDIT_2026-09-30.md).

Full defect-by-defect review with evidence: [CODE_REVIEW_0.7.0.md](CODE_REVIEW_0.7.0.md).

## You need to act on these

### 1. Existing cases cannot be upgraded in place

0.7.0 stored 64 px float frames that had already been stretched to a square.
That loss is not recoverable from the stored frames, so those cases are kept as
a separate **preprocessing generation**. They remain readable, listed and
exportable, and you can still train on them by selecting the 0.7 generation on
the training tab — but a single fit cannot mix generations.

To use the current pipeline (letterboxed, 8-bit, optional sector crop, 64/96/128
px), **re-ingest those studies from source**. The library marks each study's
generation and edge so you can see what needs redoing.

### 2. Models trained on 0.7.0 cannot be imported

The head now has 22 outputs rather than 15, and the version record carries a
third network (attention). Schema moved from `cus-browser-features-1` to
`cus-browser-features-2`, and imports of the old schema are refused rather
than silently reinterpreted. Old 0.7.0 *case* backups still restore.

### 3. Models that previously activated may now be marked exploratory

The gate is unchanged in its point-estimate condition, so nothing that passed
before now fails outright. But a model whose holdout is too small to bound
performance is labelled **exploratory**, and reading with it requires an
explicit acknowledgement each session. If your library is small, expect this.

Reaching **development-qualified** takes roughly 16 correct calls in each class
per learned finding. That is what an 80% claim means with a 95% lower bound,
and it is a statement about your dataset size, not about the model.

## Changes

### Image decoding

- NIfTI intensities are windowed on the 1st-99th percentile. 0.7.0 passed any
  volume already inside 0-255 through unscaled, so a 0-12 integer volume
  rendered across 12 of 255 levels and a float volume normalised to 0-0.3
  across 0.3 of 255. The supplied 0-255 int16 sample was unaffected; converted
  or normalised volumes were not.
- 2D-plus-time NIfTI cines are decoded. 0.7.0 rejected any 4-D volume whose
  fourth dimension was not 1, which is exactly how a cine is stored.
- Frames are letterboxed, not stretched. The supplied 733 x 494 geometry was
  previously compressed by a factor of 1.48 in one axis.
- Optional sector cropping, estimated once per source and recorded in the
  audit, declined when it would discard more than three quarters of the frame.
- Saturated-pixel screen warns about possible burned-in annotation. It measures
  brightness only and is an aid, not a de-identification guarantee.
- Frames are stored 8-bit at a selectable 64/96/128 px edge; backups pack each
  case's frames into one buffer.

### Statistics and gating

- 95% Wilson intervals on every sensitivity and specificity, shown wherever a
  score appears.
- ROC AUC and Brier score per finding.
- Operating points selected by Youden's J on an inner tuning fold split from
  the training partition and grouped by infant. The development holdout is
  never read during epoch or threshold selection.
- Three gate tiers: insufficient, exploratory, development-qualified.
- Replacement requires an infant-clustered bootstrap on the difference in mean
  balanced accuracy with a 95% lower bound above zero (2,000 resamples of whole
  infants), replacing the flat two-percentage-point comparison of point
  estimates.
- Downloadable model card per version: provenance and seed, partition
  composition, per-finding training support, preprocessing, architecture,
  threshold provenance, metrics with intervals, gate decision, and the
  outstanding-validation list.

### Learner

- Four-layer encoder with spatial average **and** maximum pooling; gated
  attention pooling across frames. 39,879 trainable parameters, against 2,543.
- Per-finding positive-class weighting, so a rare finding is no longer best
  served by predicting absent everywhere.
- Dropout 0.3, L2 1e-4, optional augmentation, recorded seed that reproduces a
  fit to within 1e-6.
- Coronal-only horizontal mirroring with hemisphere labels exchanged. Offered
  for coronal views alone because the parasagittal horizontal axis is
  anterior-posterior, where a mirror would reverse anatomy without exchanging
  hemispheres.
- Per-step frame cap bounds training cost; inference always uses every decoded
  frame.
- Active TensorFlow.js backend reported before a run, so a CPU-only fallback is
  visible rather than discovered after an hour.

### Labels

- Seven side-agnostic heads appended after the original fifteen keys, which
  keep their identity and position.
- "Not recorded in my dataset" laterality option prefills any-side findings
  only and never produces a hemisphere key. The supplied collection carries
  category labels with no hemisphere table, which under the 0.7.0 schema meant
  those 188 studies could supply no training target at all.
- Any-side heads derive from the hemispheres only when they settle it.
- Contradictory annotations are refused instead of silently resolved.

### Clinical reporting

- GMH-IVH output names the criterion that decided the grade, and flags
  discordant evidence — acute distension against a sub-threshold AHW, or an
  AHW above 6 mm with the acute criterion explicitly absent. **The grades
  themselves are unchanged**, including the strict 6 mm boundary.

### Interface

- The frame-sequence attestation is no longer pre-ticked from the decoder's own
  inference. Both confirmations start cleared on every decode.
- Per-frame attention display in the read panel, and a training loss curve.
- A **Self-tests** tab that runs the statistical and rule assertions, and a
  learning smoke test, in your own browser on your own device.
- Tab strip has `tablist`/`tab` roles with `aria-selected`; messages
  distinguish info, success and error; focus moves to the message on error;
  re-selecting the same backup file now fires.

### Build and tests

- `scripts/build_single_file.mjs` generates the single-file edition, wrapping
  each module in its own IIFE and computing both CSP hashes, then re-hashing
  the assembled document to verify them. **Do not hand-edit the bundle**: the
  policy pins the inline script hash, and an edit blocks the page with nothing
  but a console message. Rebuild with `npm run build`.
- `tests/browser_learning.test.mjs` grew from 7 to 25 tests and stays
  dependency-free. All seven original tests are unchanged and still pass.
- `tests/browser_model.test.mjs` executes the real TensorFlow.js graph on the
  CPU backend. It skips when the TensorFlow packages are absent.
- `scripts/verify_numerics.py` regenerates `tests/fixtures/stat_vectors.json`
  from statsmodels and scikit-learn, so the expectations the browser panel
  asserts have a reproducible provenance and cannot be edited to agree with a
  broken implementation.
- CI runs the Python suite, both browser suites, and a determinism check on the
  bundle.

## Verification performed

| Suite | Result |
|---|---|
| `tests/browser_learning.test.mjs` | 25 tests pass, no external dependencies |
| `tests/browser_model.test.mjs` | 6 tests pass on the CPU backend |
| `scripts/verify_numerics.py` | 43 checks against statsmodels and scikit-learn |
| Single-file bundle boot | boots under jsdom; CSP hashes independently re-verified; bundled self-tests pass |

Two faults were found by running the graph rather than reasoning about it, and
are worth recording because they would not have been caught otherwise: the
attention pooling first normalised a `[frames, 1]` tensor along axis 0, which
`tf.softmax` does not support; and the build script's own hash self-check
caught a trailing-newline mismatch that would have invalidated the pinned style
hash and shipped the page unstyled.

### Not verified

No browser-engine run of the WebCodecs MP4 path, the `ImageDecoder` animated
path, or canvas-based preprocessing on real media — jsdom provides no canvas or
WebCodecs. Those paths are unchanged in structure from 0.7.0 apart from routing
pixels through the new preprocessing context. Decode a known study and run the
**Self-tests** tab once in your own browser before trusting a fit.

No diagnostic accuracy, calibration, or subgroup performance was measured.

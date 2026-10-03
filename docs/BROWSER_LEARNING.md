# Online reader and actual image-model learning — 0.8.0

The GitHub Pages root runs the reader. No backend or API credentials are
required for this browser version. Python/Streamlit remains available for full
consensus classification, DICOM, additional codecs and larger studies.

Defects repaired in this version, with the evidence for each, are recorded in
[CODE_REVIEW_0.7.0.md](CODE_REVIEW_0.7.0.md).

## Learned model

Every decoded frame is converted to an 8-bit greyscale tensor at a selectable
64, 96 or 128 px edge, letterboxed so source aspect ratio is preserved. A
trainable CNN (12, 24, 32 and 48 filters, same-padded, with pooling between)
produces a feature map, and the spatial **average and maximum** of that map
form a 96-dimensional frame embedding. Keeping the spatial maximum is the
substantive change from 0.7.0, which globally averaged the whole frame and so
diluted any focal lesion into a 16-channel mean.

Frames are combined by gated attention pooling, concatenated with the
frame-wise maximum, giving a 192-dimensional study representation. A dense head
with dropout estimates 22 independently labeled findings: left/right GMH, IVH,
acute distension, focal periventricular echogenicity, PVE brighter than
choroid, inhomogeneous PVE and periventricular cysts; CBH; and seven
side-agnostic equivalents. Trainable parameters: 39,879, against 2,543 in
0.7.0.

Masked binary cross-entropy excludes unknown targets, with the positive term
weighted per finding by the training negative-to-positive infant ratio (clipped
to [0.25, 4]) so that a rare finding is not best served by predicting absent
everywhere. One complete study is one optimization step, so a longer clip does
not count as more independently labeled cases. A per-step frame cap bounds
training cost; **inference always uses every decoded frame**. All weights are
trained with Adam from a recorded seed, so a version is reproducible from its
own record.

This is a small CNN trained from scratch; no generic vision API, nearest-case
vote, or diagnostic ImageNet classifier is substituted. It remains a
development baseline, not a claim of clinical performance. Resolution can still
obscure small lesions. There is no automatic plane detector or validated
laterality model. The user confirms acquisition context and laterality before
labeling or estimating.

### Attention weights

The read panel ranks frames by their attention weight. These show which frames
the pooling layer relied on. They are not lesion localisation and have not been
compared against expert annotation or gaze.

### Augmentation

Brightness, contrast, shift and speckle are optional and applied per study.
Horizontal mirroring with the hemisphere labels exchanged is offered **only**
for studies labeled coronal: in a parasagittal view the horizontal axis is
anterior-posterior, so a mirror would reverse anatomy without exchanging
hemispheres.

## Category supervision and laterality

Category labels are retained as case tags. Confirmed Grade I prefills
ipsilateral GMH present / IVH absent; Grade II prefills IVH present; Grade III
additionally prefills acute distension present. No AHW value is inferred. Other
hemispheres and unrelated domains remain unknown. Normal prefills require the
expert's explicit complete-normal-examination confirmation, including
posterior-fossa views. WMI, PVHI and PHVD category names alone do not settle
all current observable findings.

New in 0.8.0: when the source dataset does not record the hemisphere, the
laterality selector offers "not recorded", which prefills the **any-side**
findings only and leaves both hemispheres unknown. This exists because the
supplied collection carries category labels with no hemisphere table (see the
accuracy audit), which under the 0.7.0 schema meant those studies could supply
no training target at all.

An any-side head is otherwise derived from the hemispheres, and only when they
settle it: either side positive makes it positive, both sides negative make it
negative, one side unknown leaves it unknown. Contradictory annotations are
refused rather than silently resolved.

## Partition and update controls

- Infants cannot cross training, development holdout, or external-test partitions.
- Duplicate source hashes cannot be added as new cases.
- Previously trained infants/sources cannot be reassigned to a holdout after removal.
- External-center identity, when supplied, cannot overlap development centers.
- Every trainable head needs two positive and two negative training infants, and
  there must be at least four training infants overall.
- A single fit cannot mix preprocessing generations or stored resolutions.
- The external partition is never used by training or the activation gate.
- Candidates are saved separately from the active version. Training or cancellation
  never changes the active model.
- Epoch and operating-point selection use an inner tuning fold split from the
  **training** partition, grouped by infant. The development holdout is never
  read during selection.
- Activation recomputes results on the current holdout. Every learned finding
  needs two positive and two negative holdout studies and at least four holdout
  infants overall.
- Imported models remain candidates until independently rechecked locally.

### Gate tiers

| Tier | Condition | What it permits |
|---|---|---|
| insufficient | point estimates below 80%, or the holdout is not adequately populated | nothing |
| exploratory | point estimates clear 80% sensitivity and specificity, but the 95% Wilson lower bound does not | reading for workflow testing, after an explicit per-session acknowledgement |
| development-qualified | every learned finding holds a 95% Wilson lower bound at or above 80% for both sensitivity and specificity | reading as the active research model |

A replacement additionally needs an infant-clustered bootstrap on the
difference in mean balanced accuracy whose 95% lower bound exceeds zero, over
2,000 resamples of whole infants. The 0.7.0 flat two-percentage-point rule is
retained as the exploratory-tier condition.

Why the tiers exist: at the documented minimum of two positive and two negative
holdout studies, a perfect score carries a 95% Wilson lower bound of 34.2%, and
a coin-flip classifier clears 80% sensitivity *and* 80% specificity about 1 time
in 16. Reaching a 0.80 lower bound takes roughly 16 correct calls in each class.
Because the gate requires every learned finding to pass simultaneously it is an
intersection of conditions and therefore conservative, so no multiplicity
adjustment is applied.

These remain exploratory development gates. Repeated use of the same holdout for
model selection can bias its measured performance; it is deliberately called a
development holdout. The independent external set must remain untouched for a
prespecified final audit. External-center testing, probability calibration,
scanner subgroups and specialist adjudication remain necessary for diagnostic
validation. Browser model versions always record `externalValidated: false`.

## Reporting

Every version exposes, per finding, sensitivity and specificity with 95% Wilson
intervals, ROC AUC, Brier score, the selected operating point and how it was
chosen. A downloadable model card records provenance (seed, dataset
fingerprint, TensorFlow.js backend), partition composition, per-finding
training support, preprocessing configuration, architecture, the gate decision
and the outstanding-validation list.

Study-level intervals are optimistic wherever one infant contributes more than
one holdout study; the model card states this, and the replacement bootstrap
clusters by infant.

## Media and persistence

Supported browser formats: PNG/JPEG/BMP, GIF/WebP with exhaustive ImageDecoder,
MP4/MOV with MP4Box + WebCodecs, single-file NIfTI-1 `.nii` / `.nii.gz`,
including 2D-plus-time cines stored on the fourth axis. Volumes with two
non-singleton candidate axes, and unsupported datatypes, are rejected. NIfTI
intensities are windowed on the 1st to 99th percentile, estimated from a
deterministic stride sample; every frame is still decoded, and only the
percentile estimate is sampled. Header spacing is not treated as verified
measurement scale. MP4 decoding checks every extracted sample against decoded
frame count and sorts presentation timestamps. Unsupported codecs fail
explicitly. Image previews are limited to eight thumbnails; all decoded frames
enter learning and inference. The browser budget rejects studies over 256
frames or sources over 256 MB.

Optional sector cropping is estimated once per source from its first frame,
since sector geometry is fixed within an acquisition, and is declined when it
would discard more than three quarters of the frame. The crop box is recorded
in the technical audit. A saturated-pixel screen on the upper band warns about
possible burned-in annotation; it measures brightness only, cannot read text or
recognise a logo, and is an aid rather than a de-identification guarantee.

Cases store 8-bit resized training frames, source hashes, technical audits,
independent labels, preprocessing configuration and partition metadata in
IndexedDB under schema `cus-browser-features-2`. Original full-resolution files
are not persisted. Versions store encoder, attention and head topology and
weights, feature schema, training lineage, seed, thresholds, timestamps,
holdout metrics and activation state. A private JSON backup packs each case's
frames into a single buffer; it contains resized frames and learned weights and
must not be published to GitHub. Each browser profile/device has its own
private library and active model. This is not an institution-wide shared
training service or managed clinical deployment.

0.7.0 cases remain readable and exportable. They were stored at 64 px with
stretched aspect under schema `cus-browser-features-1`, and are presented as a
separate preprocessing generation; a fit cannot mix generations, so to use the
current pipeline those studies must be re-ingested from source.

The initial page loads code from GitHub Pages and two integrity-pinned CDN
scripts. There is no source-media upload, telemetry endpoint or external AI
inference call. The browser must permit IndexedDB; clearing site storage
removes cases/models. Use only de-identified research media and an
appropriately governed device/profile.

## Classification boundary

Neural scores remain estimates of observable image findings. They never
populate the verified clinical form automatically. GMH-IVH in the browser is
computed only from independently verified expert findings, with Grade III
requiring acute ipsilateral distension and calibrated AHW strictly above 6 mm.
The paper's rules are fixed; learning cannot rewrite medical definitions.

0.8.0 additionally reports the criterion that decided each grade, and flags
discordant evidence — acute distension reported against a sub-threshold AHW, or
an AHW above 6 mm with the acute criterion explicitly absent — rather than
silently grading on one of them. The grades themselves are unchanged.

Full PVHI/WMI/CBH/PHVD and normal-examination classification remain in the
Python expert workflow. Serial duration, verified measurements and complete
coverage cannot be invented by a CNN.

## Verification

Run `node tests/browser_learning.test.mjs` for the dependency-free suite:
interval statistics against values generated from statsmodels and
scikit-learn, promotion gating, the seeded shuffle, label algebra, the fixed
paper rules, NIfTI decoding and windowing, letterbox geometry, sector-crop
refusal, and a check that the committed single-file bundle still matches the
modules.

Run `npm install && node tests/browser_model.test.mjs` to execute the actual
TensorFlow.js graph on the CPU backend: encoder shapes at all three
resolutions, parameter count, a real fit, attention normalisation across
frames, inference determinism, save/reload fidelity, and seeded
reproducibility. This file exists because shape reasoning is not sufficient —
the attention pooling initially normalised along an axis `tf.softmax` does not
support, which only running the graph reveals.

The reader also ships a **Self-tests** tab. It runs the same statistical and
rule assertions in the user's own browser, and a learning smoke test that fits
a few epochs on synthetic noise and confirms the model graph executes, that
attention weights sum to one, and that weights survive serialisation on that
device. No clinical performance estimate is derived from any synthetic or
format check.

Implementation references: [TensorFlow.js training](https://www.tensorflow.org/js/guide/train_models),
[model persistence](https://www.tensorflow.org/js/guide/save_load),
[WebCodecs](https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API/Using_the_WebCodecs_API),
[attention-based deep MIL](https://arxiv.org/abs/1802.04712),
[Wilson score interval](https://doi.org/10.1080/01621459.1927.10502953).

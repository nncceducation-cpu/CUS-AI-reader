# Online reader and actual image-model learning — 0.7.0

The GitHub Pages root now runs the reader. No backend or API credentials are
required for this browser version. Python/Streamlit remains available for full
consensus classification, DICOM, additional codecs and larger studies.

## Learned model

Every decoded frame is converted to a 64 × 64 grayscale tensor. A trainable CNN
(8-filter convolution, pooling, 16-filter convolution, global-average pooling)
produces 16 frame features. Their temporal mean and maximum form a 32-dimensional
study representation. A trainable dense head estimates 15 independently labeled
findings: left/right GMH, IVH, acute distension, focal periventricular echogenicity,
PVE brighter than choroid, inhomogeneous PVE, periventricular cysts, and CBH.

Masked binary cross-entropy excludes unknown targets. One complete study is one
optimization step, so a longer clip does not count as more independently labeled
cases. Both CNN and dense-head weights are trained with Adam. This is a small CNN
trained from scratch; no generic vision API, nearest-case vote, or diagnostic
ImageNet classifier is substituted. This architecture is a development baseline,
not a claim of clinical performance. The low resolution can obscure small lesions.
There is no automatic plane detector or validated laterality model in this version.
The user confirms acquisition context and laterality before labeling or estimating.

Category labels are retained as case tags. Confirmed Grade I prefills ipsilateral
GMH present / IVH absent; Grade II prefills IVH present; Grade III additionally
prefills acute distension present. No AHW value is inferred. Other hemispheres and
unrelated domains remain unknown. Normal prefills require the expert's explicit
complete-normal-examination confirmation, including posterior-fossa views. WMI,
PVHI and PHVD category names alone do not settle all current observable findings.

## Partition and update controls

- Infants cannot cross training, development holdout, or external-test partitions.
- Duplicate source hashes cannot be added as new cases.
- Previously trained infants/sources cannot be reassigned to a holdout after removal.
- External-center identity, when supplied, cannot overlap development centers.
- Every trainable head needs two positive and two negative training infants, and
  there must be at least four training infants overall.
- The external partition is never used by training or the activation gate.
- Candidates are saved separately from the active version. Training or cancellation
  never changes the active model.
- Activation recomputes results on the current holdout. Every learned finding
  needs two positive and two negative holdout studies, at least four holdout infants
  overall, and sensitivity and specificity of at least 0.80 at the fixed 0.50
  feature-score threshold. A replacement also needs at least 0.02 improvement in
  balanced accuracy versus the active model on matched outputs and the same holdout.
- Imported models remain candidates until independently rechecked locally.

These are exploratory development gates. Small-sample point estimates are not
adequate clinical evidence. Repeated use of the same holdout for model selection
can bias its measured performance; it is deliberately called a development holdout.
The independent external set must remain untouched for a prespecified final audit.
External-center testing, confidence intervals, probability calibration, scanner
subgroups and specialist adjudication remain necessary for diagnostic validation.
Browser model versions always record `externalValidated: false`.

## Media and persistence

Supported browser formats: PNG/JPEG/BMP, GIF/WebP with exhaustive ImageDecoder,
MP4/MOV with MP4Box + WebCodecs, single-file NIfTI-1 `.nii` / `.nii.gz`.
Stored NIfTI third-axis order is retained; nonsingleton fourth axes and unsupported
datatypes are rejected. Header spacing is not treated as verified measurement scale.
MP4 decoding checks every extracted sample against decoded frame count and sorts
presentation timestamps. Unsupported codecs fail explicitly. Image previews are
limited to eight thumbnails; all decoded frames enter learning and inference.
The browser budget rejects studies over 256 frames or sources over 256 MB.

Cases store resized training frames, source hashes, technical audits, independent
labels and partition metadata in IndexedDB. Original full-resolution files are not
persisted. Versions store CNN/head topology and weights, feature schema, training
lineage, timestamps, holdout metrics, and activation state. A private JSON backup
contains resized frames and learned weights; it must not be published to GitHub.
Each browser profile/device has its own private library and active model. This is
not an institution-wide shared training service or managed clinical deployment.

The initial page loads code from GitHub Pages and two integrity-pinned CDN scripts.
There is no source-media upload, telemetry endpoint or external AI inference call.
The browser must permit IndexedDB; clearing site storage removes cases/models.
Use only de-identified research media and an appropriately governed device/profile.

## Classification boundary

Neural scores remain estimates of observable image findings. They never populate
the verified clinical form automatically. GMH-IVH in the browser is computed only
from independently verified expert findings, with Grade III requiring acute
ipsilateral distension and calibrated AHW strictly above 6 mm. The paper's rules
are fixed; learning cannot rewrite medical definitions. Full PVHI/WMI/CBH/PHVD and
normal-examination classification remain in the Python expert workflow. Serial
duration, verified measurements and complete coverage cannot be invented by a CNN.

## Verification

Node tests cover partition leakage, duplicates, unknown-label masks, category
mapping, abstention/promotion gates, clinical boundaries and NIfTI truncation/order.
An isolated Chrome test decoded every frame of a synthetic 12-frame H.264 MP4 and
the supplied 22-frame NIfTI, saved/reloaded a case, trained the CNN on synthetic
data, serialized/restored weights, and confirmed unchanged predictions. No clinical
performance estimate was derived from these synthetic or format checks.

Implementation references: [TensorFlow.js training](https://www.tensorflow.org/js/guide/train_models),
[model persistence](https://www.tensorflow.org/js/guide/save_load),
[WebCodecs](https://developer.mozilla.org/en-US/docs/Web/API/WebCodecs_API/Using_the_WebCodecs_API).

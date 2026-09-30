# CUS reader accuracy audit — 30 September 2026

## Outcome

Version 0.6.1 repairs classification and data-ingestion defects. It does **not**
establish an accurate diagnostic model. The bundled ImageNet MobileNet encoder
and 15-study / 8-infant similarity reference fit are not trained lesion detectors.
Their neighbor votes no longer manufacture imaging features or consensus grades.
The pilot remains available as clearly unvalidated reference retrieval.
No new diagnostic weights have been trained, no prospective accuracy has been
measured, and no online application has been deployed by this audit.

## Findings and repairs

- Similarity grades previously became presumed hemorrhage, acute distension and
  AHW threshold evidence. That conversion has been disconnected.
- Upstream abstention and incomplete decoding now withhold every AI domain.
  Withheld results cannot remain plausible grades in displays or agreement counts.
- Missing hemorrhage compartments and uncertain anatomical localization remain
  unknown; present bleeding does not prove preceding hemorrhage.
- One accepted frame does not prove a complete coronal, sagittal or mastoid sweep.
- Grade II is recognized when an acute Grade III criterion is explicitly absent;
  unresolved acute distension and AHW cannot release a definite Grade III.
- Inhomogeneous periventricular echogenicity is represented separately from
  brightness relative to choroid plexus in the evidence and expert form.
- DICOM resizing updates millimetres per working pixel. Native 8-bit brightness
  is preserved. Narrow monochrome multiframe objects are not mistaken for RGB.
- Invalid or missing model rows, invalid probabilities, altered weights and
  paths outside the model directory are rejected.
- Frame ceilings reject rather than silently sample clips, GIFs or DICOM.
- NIfTI imports decode every third-axis frame in stored order with a common
  display scale. Unverified modality, axis interpretation and header spacing
  remain explicit. Ambiguous nonsingleton fourth dimensions are rejected.
- Dataset auditing detects infant overlap, duplicate-media overlap and reuse
  of an external-test center in development. Passing it is not proof of quality.
- The corrections tab is identified as calibration, not image-model retraining.

## Supplied cases

The supplied Drive folder inventory contains **268 files, all `.nii.gz`**, totaling
14,014,122,595 bytes. The reader previously did not support this format.

| Collection | Files |
|---|---:|
| Common preterm injury / abnormalities | 92 |
| Preterm test cases | 80 |
| Term injury / malformations | 48 |
| Mastoid window | 18 |
| PHVD | 30 |

There are six files in the normal category. The owner confirms that the category
collections are labeled and the test cases are unlabeled: **188 category-labeled
files and 80 unlabeled test-case files**. These supplied labels are retained as
category supervision. No separate table of hemisphere-specific grades,
measurements or infant identities was found in the recursive inventory.
Names include MRI, MR and malformation examples, which cannot be assumed to be
ultrasound within the preterm consensus taxonomy. Repeated filenames and matching
sizes across teaching and test collections suggest reused material; byte hashes
are required before declaring duplicates. Serial naming inconsistencies also
require an infant/time-point crosswalk. File count is not independent infant count.

One normal coronal example was downloaded privately to check format support:
shape 733 × 494 × 22, signed 16-bit stored values ranging 0–255. All 22 frames
decode. This is a format check, not a diagnostic evaluation. Source media, private
inventory and newly derived weights are excluded from the source distribution.

## Classification contract

Use the [Canadian consensus paper](https://www.frontiersin.org/journals/pediatrics/articles/10.3389/fped.2021.618236/full)
as the rule source. GMH-IVH and PVHI require hemisphere-specific outputs; PVHI
is a separate entity rather than a universal legacy Grade IV. Acute Grade III
requires ipsilateral distension and AHW strictly greater than 6 mm. WMI duration
and cystic evolution require serial evidence. PHVD requires preceding hemorrhage,
timing and age-specific VI plus AHW thresholds. CBH requires posterior-fossa
assessment and size/extent. Injuries may coexist; normal versus one exclusive
injury label is an inadequate output design. An isolated normal-looking frame
cannot establish a complete normal examination.

## Primary research reviewed

| Study | Evidence | Relevance and limit |
|---|---|---|
| [Kim 2022](https://pubmed.ncbi.nlm.nih.gov/35064324/) | 400 infants; GMH classifier test accuracy 0.875 | Narrow sagittal GMH task, not the entire consensus taxonomy. |
| [Peng 2025](https://link.springer.com/article/10.1007/s00247-025-06327-x) | 1,060 participants; prospective two-center set of 287, AUC 0.961 and accuracy 0.89 | Demonstrates specialist-model potential; does not validate this reader or every consensus domain. |
| [Lin 2025](https://www.nature.com/articles/s41467-025-63096-9) | 8,757 images; internal/external video AUC 0.982 / 0.944 | Multiview model and standard-plane detection; severe/nonsevere screening is not normal/all-injury classification. |
| [Ahmad / Afifi 2024 online](https://pubmed.ncbi.nlm.nih.gov/39212671/) | 4,180 images from 538 very preterm infants; normal/abnormal AUC 0.86 | Canadian research with uncertainty handling, but not full laterality/serial grading. |
| [Zhu 2023](https://www.frontiersin.org/journals/pediatrics/articles/10.3389/fped.2023.1144952/full) | 807 images from 158 infants, including 32 WMI; AUC 0.863, segmentation Dice 0.78 | Useful WMI/segmentation direction; does not establish all serial WMI grades. |
| [Ibrahim 2024](https://www.mdpi.com/1424-8220/24/21/7052) | 586 infants; reported mAP50 0.979 | Object-detection mAP is not examination-level diagnostic accuracy; legacy grades require careful mapping. |
| [Pham 2025](https://pubmed.ncbi.nlm.nih.gov/40376373/) | 35 infants; generic ChatGPT IVH sensitivity 75%, four of 16 IVH cases missed | Does not support replacing the reader with generic vision prompting. |

The most useful inspected open-code candidate is
[NCLS](https://github.com/Je1zzz/Neonatal_cerebral_lesions_screening_NCLS), with an
MIT license and separate author-linked weights. Its plane detector and specialized
encoder are candidates for evaluation and adaptation. Its severe/nonsevere output
must not be relabeled as normal/abnormal, and its reported performance does not
transfer to this dataset. No NCLS checkpoint was installed or validated here.

## Work needed to establish accuracy

1. Preserve the 188 supplied category labels. Reconcile source identity and acquisition dates, confirm ultrasound modality,
   identify cine versus spatial-volume axes, and hash every source before splitting.
2. Map existing labels to the paper's domains without inventing laterality or
   negative findings in other domains. Have two independent experts annotate
   missing domain details and the unlabeled test cases, and adjudicate
   disagreements. Record normal studies, left/right GMH-IVH and PVHI, WMI pattern
   and persistence, CBH size/extent, AHW/VI with verified calibration, postnatal
   and gestational ages, preceding hemorrhage and complete view coverage. Unknown
   and unavailable domains must be explicit rather than treated as negative.
3. Keep all scans from each infant within one partition. Reserve calibration and
   untouched internal testing, plus an independent center for external testing.
   Folder test labels alone do not prove independence from teaching material.
4. Train anatomy/plane and lesion/measurement components with temporal evidence;
   use the deterministic paper rules for final multi-domain classification.
   Fine-tuning requires representative normal and each injury category; rare
   categories may require additional institutions and cannot be solved by thresholds.
5. Measure per-domain and examination sensitivity, specificity, predictive values,
   ROC/PR measures where appropriate, calibration, abstention/coverage, agreement
   and confidence intervals clustered by infant. Include failure analysis by
   center, scanner, age, view quality, category and laterality. Compare against
   the existing reader on exactly the same untouched cases.
6. Decide prespecified clinical acceptance criteria with expert users before
   prospective blinded evaluation. No unsupported universal accuracy target is
   claimed by this update.

## Verification and delivery

The automated suite passed 106 tests. The Streamlit application opened with all
five tabs without exceptions. The real NIfTI sample decoded every stored frame.
These checks establish software behavior, not diagnostic sensitivity or specificity.
The supplied source ZIP can be installed using `requirements.txt` and launched
with `python -m streamlit run app.py`. It is not a rebuilt portable executable.

The immediate missing inputs for valid full-domain evaluation are the test-case
answer key, infant/scan crosswalk and the finer domain annotations not encoded by
category names. Existing category labels are usable partial supervision, rather
than substitutes for all hemisphere, measurement and longitudinal outputs.

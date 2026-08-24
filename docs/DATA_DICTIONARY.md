# Data and annotation dictionary

## Identifiers and dates

| Field | Type | Required | Definition |
|---|---|---:|---|
| study_code | string | yes | Project-specific de-identified examination code. Never use medical record number. |
| infant_code | string | development dataset | Stable de-identified infant code linking serial studies. |
| center_code | categorical | development dataset | De-identified center label used for external and subgroup evaluation. |
| scanner_code | categorical | yes | Manufacturer and model or an approved de-identified mapping. |
| acquisition_datetime_shifted | datetime | development dataset | Consistently date-shifted timestamp that preserves intervals. |
| postnatal_age_days | numeric | yes | Age at the examination, in days. |
| gestational_age_weeks | numeric | yes | Gestational age at birth, in completed weeks plus decimal fraction. |
| postmenstrual_age_weeks | numeric | PHVD | Age used to select ventricular reference charts. |

## Media and quality

| Field | Type | Levels or unit |
|---|---|---|
| media_type | categorical | DICOM single frame, DICOM multiframe, image, video |
| window | categorical | anterior, mastoid, posterior, unknown |
| orientation | categorical | coronal, sagittal midline, parasagittal left, parasagittal right, axial or other, unknown |
| plane | categorical | frontal horns or foramen of Monro, ventricular body, trigone, occipital horns, third or fourth ventricle, cerebellum, other |
| diagnostic_quality | ordinal | adequate, limited but interpretable, nondiagnostic |
| quality_reason | multilabel | blur, low contrast, depth, gain, crop, artifact, missing anatomy, compression, unknown laterality |
| pixel_spacing_mm | pair numeric | row and column spacing in mm per pixel |
| frame_index | integer | position in source object or clip |
| source_frame_count | integer | total frames reported by the source container |
| decoded_frame_count | integer | number of frames decoded sequentially to the end of the source |
| all_frames_processed | boolean | whether decoded and reported frame counts agree, or the stream had no reliable count and reached its end |

## Imaging features by hemisphere

| Field | Type | Definition |
|---|---|---|
| hemorrhage_present | yes, no, unknown | Hemorrhage within or around germinal matrix or in ventricular system. |
| confined_to_germinal_matrix | yes, no, unknown | No blood in lateral ventricle. |
| intraventricular_blood | yes, no, unknown | Blood in lateral ventricle or on choroid plexus. |
| ventricular_distension | yes, no, unknown | Acute ipsilateral distension caused by intraventricular clot. |
| ahw_mm | numeric | Maximum diagonal anterior horn width on correct coronal plane. |
| vi_mm | numeric | Falx to lateral wall of anterior horn on correct coronal plane. |
| adjacent_periventricular_echogenicity | yes, no, unknown | Focal echogenicity adjacent to ipsilateral GMH-IVH. |
| echogenicity_brighter_than_choroid | yes, no, unknown | White matter echogenicity above choroid plexus reference. |
| cystic_change | categorical | none, localized, extensive periventricular, deep or subcortical, porencephalic, unknown. |
| clinician_verified | boolean | Expert confirmed source evidence rather than accepting model output. |

## Study-level serial domains

| Field | Type | Definition |
|---|---|---|
| maximum_first_week_grade_left | ordinal | Negative, I, II, III, indeterminate. |
| maximum_first_week_grade_right | ordinal | Negative, I, II, III, indeterminate. |
| pvhi_left | categorical | Present, absent, indeterminate. |
| pvhi_right | categorical | Present, absent, indeterminate. |
| wmi_pattern | categorical | None, PVE under 7 days, Grade 1, Grade 2, Grade 3, Grade 4, indeterminate. |
| cerebellar_hemorrhage | categorical | None, punctate, limited, large, indeterminate. |
| prior_gmh_ivh | yes, no, unknown | Required before ventricular dilatation is labeled PHVD. |
| vi_above_97th | yes, no, unknown | Based on postmenstrual age reference chart. |
| vi_above_97th_plus_4mm | yes, no, unknown | Severe PHVD threshold. |
| phvd | categorical | None, moderate, severe, indeterminate, not PHVD. |

## Model record

Every inference record must include:

- model ID, semantic version, commit, and weight checksum
- manifest checksum and consensus rule version
- preprocessing version
- input file hash without storing the source file in the exported report
- plane and quality probabilities
- feature probabilities by frame, plane, and hemisphere
- the processing result for every decoded frame, including ambiguous plane assignments
- calibration version and locked thresholds
- abstention status and reasons
- clinician acceptance or correction for every suggested feature
- final classification and all limitations

## Annotation quality control

Use explicit `unknown` rather than converting missing evidence to `no`. Store the two initial reader labels, adjudicated label, reader role, and review timestamp. For masks and calipers, store each reader's annotation and the adjudicated result. Calculate agreement before adjudication.

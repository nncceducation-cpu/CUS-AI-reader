# AI-to-consensus method

## Model target

The system does not train a direct black-box Canadian grade classifier. It estimates observable imaging features, laterality, plane adequacy, and ventricular thresholds. A versioned deterministic engine then applies the Canadian consensus rules. This structure makes each grade traceable to the model outputs and supports expert correction at the feature level.

The required model heads are:

- coronal, sagittal or parasagittal, posterior fossa, other, and indeterminate plane
- left and right hemorrhage presence
- confinement to germinal matrix and intraventricular extension
- acute ventricular distension
- AHW above 6 mm and above 10 mm
- focal periventricular echogenicity and echogenicity relative to choroid plexus
- serial WMI categories
- CBH categories
- VI above the 97th centile and above the 97th centile plus 4 mm

Continuous AHW and VI landmark models should be added when native pixel spacing or a validated scale is available. Binary threshold heads do not replace saved caliper measurements in the expert reference set.

## Inference

Every decoded frame enters the model in source order. Plane probabilities are stored for every frame. Lesion probabilities are aggregated only within accepted planes. A probability within the locked decision margin is converted to unknown, not to no. Missing required heads, missing planes, incomplete frame decoding, uncertain feature decisions, or missing serial evidence force abstention from a final grade. Partial research outputs remain visible for error analysis.

## Training and evaluation

All examinations from an infant remain in one partition. The eight-infant pilot is reserved for annotation and workflow testing. Model development requires a larger adjudicated set with enough affected infants for each target and an external site held out from all training, tuning, and threshold selection. The primary endpoint should be locked before analysis. The current validation protocol recommends per-infant sensitivity for severe preterm brain injury with confidence intervals and center-clustered resampling.

The NCLS system is a candidate starting backbone for standard-view extraction and lesion screening. Its published severe or non-severe output is not equivalent to the Canadian consensus categories. Direct reuse therefore requires a separate adapter, confirmation of weight redistribution terms, fine-tuning on the Canadian feature targets, calibration, and external validation.

## Agreement

The app reports exact agreement for the current study across left and right GMH-IVH, PVHI, WMI, CBH, PHVD, and the severe-injury flag. Cohen kappa is not calculated across unlike clinical domains. Domain-specific kappa is appropriate only after multiple independently graded studies have accumulated.

## Sources

- Mohammad K, Scott JN, Leijser LM, et al. Front Pediatr. 2021;9:618236. doi:10.3389/fped.2021.618236.
- Zhang J, et al. Deep learning approach for screening neonatal cerebral lesions on ultrasound in China. Nat Commun. 2025. doi:10.1038/s41467-025-63096-9.


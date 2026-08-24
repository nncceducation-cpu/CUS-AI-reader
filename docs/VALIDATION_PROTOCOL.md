# Development and validation protocol

## Intended use under evaluation

The candidate system reviews de-identified neonatal cranial ultrasound examinations from preterm infants. It identifies whether required views are present, localizes specified imaging features, measures ventricles when scale is valid, and proposes feature probabilities for clinician review. A deterministic engine then maps clinician-verified features to the Canadian consensus categories.

The first evaluation is retrospective and silent. No output is returned to the clinical team. A prospective silent study follows only after external retrospective performance is acceptable. Clinical use requires a separate regulatory and implementation program.

## Population

Include infants born at 31 completed weeks of gestation or earlier who undergo routine cranial ultrasound, plus preterm infants outside that range who undergo cUS for clinical instability. Predefine whether term infants and congenital brain malformations are outside the intended population.

Preserve difficult examinations, portable scanners, artifacts, postoperative scans, and studies with overlapping lesions. Do not exclude cases because the algorithm is expected to fail on them. Record the reason for every exclusion.

## Index test

Freeze the complete software version before evaluation, including preprocessing, plane gate, thresholds, calibration, ensemble, abstention rules, measurement code, and consensus rule version. Do not update weights or thresholds during the evaluation period.

## Reference standard

Two readers independently review the full examination and relevant serial studies using the locked annotation manual. At least one reader should be a pediatric radiologist or neuroradiologist and one should be a neonatologist with formal neonatal neuroimaging expertise. Disagreements are resolved by an independent third expert without access to algorithm output.

MRI can provide supportive evidence for subtle hemorrhage, posterior fossa lesions, or other injury when performed clinically. MRI availability must not determine cohort inclusion. Record whether MRI changed the adjudicated cUS label.

## Annotation units

Annotate at four levels:

1. Frame: plane, orientation, quality, laterality, visible structures, lesion masks, and measurement landmarks.
2. Examination: left and right features, WMI, CBH, measurements, adequacy, and abstention reason.
3. Infant: maximum first-week GMH-IVH grade, serial evolution, PHVD, and final consensus categories.
4. Outcome linkage: optional clinical and neurodevelopmental outcomes under a separately approved protocol.

## Data partitioning

Split by infant before any frame extraction. All images and serial studies from one infant stay in one partition. Use:

- development centers for training and tuning
- a later time period from at least one development center for temporal evaluation
- one or more unseen centers, scanners, and clinical teams for external evaluation

If the same infant is transferred between sites, link records before splitting. Do not place augmented or synthetic versions of an evaluation image in development data.

## Primary endpoints

Lock one primary endpoint before data analysis. The recommended first endpoint is per-infant sensitivity for severe preterm brain injury on the complete examination. Report the Wilson 95% confidence interval and a center-clustered bootstrap interval.

The sample size should be based on the desired confidence interval width. For orientation, if expected sensitivity is 0.90 and the desired two-sided 95% interval half-width is 0.05, a simple binomial approximation requires about 139 affected infants. Increase this for center clustering, exclusions, subgroup evaluation, and lower prevalence of individual lesions. Calculate the final number in R before data lock.

## Secondary endpoints

### Classification

- sensitivity, specificity, positive and negative predictive values with 95% confidence intervals
- one-versus-rest AUC and precision-recall AUC for each grade
- macro F1 and balanced accuracy
- weighted kappa for ordered GMH-IVH grade
- left-right laterality accuracy
- false-negative rate for Grade III, PVHI, cystic WMI, severe PHVD, and large CBH

### Calibration and abstention

- calibration intercept and slope
- Brier score and calibration plot
- coverage versus risk curve for the abstention policy
- error rate among accepted studies and among all studies
- frequency and cause of abstention by center, scanner, gestational age, and diagnosis

### Segmentation and measurement

- Dice and intersection over union
- 95th percentile Hausdorff distance
- AHW and VI mean absolute error in millimeters
- intraclass correlation coefficient
- Bland-Altman bias and 95% limits of agreement
- proportion of measurements that are technically valid

### Workflow

- time per examination
- change in reader accuracy and reading time in a crossover multi-reader multi-case study
- proportion of model suggestions changed by the clinician
- failure-recovery rate and usability errors

## Subgroups

Report performance with confidence intervals by:

- gestational age and birth weight groups
- postnatal age and scan epoch
- center and scanner manufacturer or model
- anterior versus mastoid window
- sex
- unilateral versus bilateral injury
- image quality group
- presence of multiple simultaneous lesions
- compressed export versus native DICOM

Do not claim fairness from absence of a statistically significant interaction. Report effect estimates and precision.

## Missing data

Missing views and missing scale are properties of the index test workflow, not values to impute. They should trigger the predefined abstention or measurement-disabled state. Report missing clinical variables separately. Do not use outcome-informed imputation.

## Error analysis

Review every false negative for severe injury and a stratified sample of other errors. Record whether the cause was acquisition, plane gate, localization, measurement, temporal aggregation, thresholding, or rule mapping. Review without changing the locked evaluation model. Proposed changes belong to a later version and a new evaluation.

## Reporting

Use STARD-AI for diagnostic accuracy, CLAIM 2024 for imaging AI methods, and TRIPOD+AI if a prognostic or outcome model is added. Publish the protocol, versioned code, model card, data flow, threshold rationale, and a data-access statement.

Relevant guidance:

- [STARD-AI, Nature Medicine 2025](https://doi.org/10.1038/s41591-025-03953-8)
- [CLAIM 2024 update](https://doi.org/10.1148/ryai.240300)
- [TRIPOD+AI, BMJ 2024](https://doi.org/10.1136/bmj-2023-078378)
- [Health Canada pre-market guidance for ML-enabled medical devices](https://www.canada.ca/en/health-canada/services/drugs-health-products/medical-devices/application-information/guidance-documents/pre-market-guidance-machine-learning-enabled-medical-devices.html)
- [Health Canada, FDA, and MHRA Good Machine Learning Practice principles](https://www.canada.ca/en/health-canada/services/drugs-health-products/medical-devices/good-machine-learning-practice-medical-device-development.html)

## Release gates

A multidisciplinary steering group must lock numerical gates before external evaluation. At minimum, require:

- complete external-site evaluation with the prespecified primary endpoint
- no unresolved critical false-negative pattern
- calibration and abstention performance within locked limits
- measurement agreement within clinically accepted limits
- acceptable subgroup performance or a narrowed intended population
- successful privacy, security, accessibility, and human-factors review
- documented model and data provenance
- regulatory determination and institutional approval for the intended deployment


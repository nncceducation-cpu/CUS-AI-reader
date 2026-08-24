# Algorithm review and selected approach

Version 1.0, evidence searched through 2026-08-23

## Decision

The preferred system is a feature-first, multi-task pipeline. It should not train one network to jump directly from an arbitrary image to a final diagnosis. Neonatal cranial ultrasound classification depends on plane, laterality, measurement, timing, and the evolution of findings across a study. The final Canadian consensus category should therefore be produced by a deterministic rule engine from visible and measurable features.

The production target has five model components:

1. Technical quality, standard-plane, window, and laterality gate.
2. Hemorrhage and white-matter feature detection with pixel-level localization.
3. Ventricular and lesion segmentation, followed by geometry-based measurement when DICOM scale is available.
4. Clip and study-level aggregation across frames and planes, with temporal smoothing.
5. Calibrated uncertainty and abstention before the Canadian consensus rule engine runs.

No validated public neonatal cranial ultrasound pathology weights were found. No open dataset found in this review can train and externally validate all consensus domains. For that reason, this repository includes the inference contract but does not contain diagnostic weights.

## Closest neonatal-specific evidence

| Study | Task and data | Reported performance | Interpretation for this project |
|---|---|---|---|
| Peng et al, 2025 | Periventricular-IVH detection and grading; 1,060 participants from two hospitals; retrospective development and prospective two-center validation; CNN with convolutional block attention | Prospective validation AUC 0.961; accuracy 0.89 | Closest published study to the intended task. The model still does not cover the full Canadian pathway, serial WMI, PHVD measurement, or all posterior fossa findings. |
| Kim et al, 2022 | Binary germinal matrix hemorrhage detection from one key sagittal image; 400 ultrasound examinations; transfer learning and augmentation | Validation AUC 0.92; test accuracy 0.875 | Supports a dedicated caudothalamic-groove detector, but not study-level grading. |
| Ibrahim et al, 2024 | Five-class GMH grading with ResNet and YOLOv8; 586 infants from one source | YOLOv8 mAP50 0.979 and mAP50-95 0.724 | The high reported metric needs patient-level, site-level, and terminology review before reuse. Direct five-class prediction is less traceable than feature extraction plus rules. |
| Tabrizi et al, 2018 | Automated 2D ventricle segmentation in 60 premature neonates using brain localization, fuzzy c-means, phase congruency, and active contour | Dice 0.80 plus or minus 0.12; PHH outcome accuracy 83% | Shows that ventricular geometry can be extracted, but the cohort is small and the method predates current encoder-decoder models. |
| Benavente-Fernandez et al, 2021 | 3D ventricular volume segmentation in PHVD; 152 scans from 10 preterm infants; 2D CNN applied to 3D ultrasound | Dice 0.80; ICC 0.944 | Supports volumetric measurement when 3D acquisition is available. The patient count is too small for broad deployment claims. |
| Szentimrey et al, 2024 | Semi-supervised neonatal ventricular segmentation; 887 3D scans from 47 patients, including 87 labeled scans; 3D U-Net with shape autoencoder and adversarial loss | Best reported Dice 81.2%; mean volume difference 0.09 cm3 | Strong method for limited labels. The shape constraint is well suited to low-contrast ventricular boundaries. |
| Jiao et al, USFM, 2024 | Self-supervised ultrasound foundation encoder trained on 2,187,915 multi-organ, multicenter, multidevice images | Improved label efficiency across segmentation and classification tasks | Good initialization candidate, but not evidence of neonatal cranial diagnostic validity. It requires local fine-tuning and external evaluation. |
| Pham et al, 2025 | General-purpose GPT-4o interpretation in 35 very preterm infants | Sensitivity 75%; specificity 84.2%; AUC 0.796 | Four of 16 affected infants were false negatives. A general-purpose vision-language model should not be the diagnostic core. |

## Selected model design

### 1. Input and de-identification

Use DICOM as the preferred source because it can preserve frame timing and pixel spacing. Decode cine clips without altering the original. Retain only technical acquisition fields needed for analysis. Burned-in text requires a separate de-identification process before data enter a development environment.

JPEG, PNG, and compressed clips can be reviewed, but automated millimeter measurement must be disabled unless scale has been validated.

### 2. Quality and plane model

Train a multi-task plane model to predict:

- anterior versus mastoid or posterior fontanel window
- coronal, sagittal, and parasagittal orientation
- named standard plane, including frontal horns at foramen of Monro, ventricular body, trigone, occipital horns, midline sagittal, bilateral parasagittal ventricles, and posterior fossa views
- left and right orientation
- diagnostic-quality score and reasons for rejection
- presence of calipers, annotations, or a non-ultrasound image

Use an ultrasound-pretrained encoder such as USFM as one comparator. Also train a site-specific self-supervised encoder on de-identified unlabeled clips. Compare with EfficientNetV2, ConvNeXt, and Swin Transformer baselines. Select on external-site calibration and sensitivity, not internal accuracy alone.

### 3. Feature localization and segmentation

Use separate or shared encoder-decoder heads for:

- germinal matrix clot
- intraventricular clot distinct from choroid plexus
- left and right lateral ventricles
- third and fourth ventricles
- periventricular echogenicity and cystic lesions
- cerebellar hemorrhage
- reference structures required for plane validation and measurement

For 2D tasks, compare nnU-Net v2, U-Net with an ultrasound-pretrained encoder, and a transformer decoder. For 3D ventricular data, compare 3D nnU-Net with the shape-encoding semi-supervised method described by Szentimrey et al. Segmentation provides localization and measurement, which is preferable to a class activation map that is not trained as a lesion boundary.

### 4. Clip and study aggregation

Run plane and feature inference on every decodable cine frame in sequence. Display thumbnails can be limited, but diagnostic processing must not use that preview limit. Do not classify each frame as an independent infant. Retain every per-frame output, mark ambiguous planes, then aggregate accepted frame features with one of these methods:

- attention-based multiple-instance learning for small datasets
- temporal convolution for short ordered clips
- a video transformer only after the dataset is large enough to avoid overfitting

Apply temporal smoothing so a single noisy frame cannot determine a study label. Aggregate by coronal or sagittal plane and hemisphere before producing study-level evidence. Require accepted frames from both plane families before the model can claim complete anterior-fontanel coverage. The worst verified grade in the first postnatal week is used for initial GMH-IVH classification, consistent with the consensus paper.

### 5. Measurements

Derive AHW and VI from segmented ventricular borders on the validated coronal plane at the level of the foramen of Monro. Convert pixels to millimeters only from verified DICOM spacing or an approved calibration object. Show the contour and caliper endpoints to the reviewer. If the ependymal boundary is obscured, the system should abstain rather than invent a border.

Measure PHVD on serial studies. Acute ventricular distension that accompanies clot in the first week belongs to grade III GMH-IVH. Later ventricular enlargement after GMH-IVH belongs to the PHVD pathway.

### 6. Uncertainty and abstention

Use patient-level cross-validation only during development. For release candidates, freeze the model and calibrate probabilities on a separate calibration set using temperature scaling or isotonic regression. Compare a small ensemble with deep ensembles or Monte Carlo dropout. Add an abstention rule for:

- missing required views
- out-of-distribution scanner or acquisition settings
- low technical quality
- model disagreement
- probability close to a locked decision threshold
- absent pixel spacing when a measurement is required
- disagreement between detected anatomy and predicted plane

All abstentions must be visible in the exported record.

### 7. Rule engine

The model predicts features. The rule engine in `cus_ai/clinical.py` maps verified features to the Canadian categories independently for the left and right hemispheres. This separation permits unit testing, versioning, and direct comparison with the HUS Diagnostic decision tree.

## Methods that were not selected as the primary core

- A general-purpose vision-language model alone, because neonatal validation is small and false-negative risk is too high.
- A single still-image classifier for a complete study, because WMI, PVHI evolution, PHVD, and some posterior fossa lesions require multiple planes or serial imaging.
- Direct end-to-end grade prediction without feature localization, because errors cannot be traced to the consensus criteria.
- Random frame-level splitting, because frames from the same infant can leak across development and evaluation data.
- Millimeter measurement from screenshots without verified scale.
- Synthetic images in the final evaluation set.

## Primary sources

1. Mohammad K, Scott JN, Leijser LM, et al. Consensus Approach for Standardizing the Screening and Classification of Preterm Brain Injury Diagnosed With Cranial Ultrasound: A Canadian Perspective. Front Pediatr. 2021;9:618236. [doi:10.3389/fped.2021.618236](https://doi.org/10.3389/fped.2021.618236).
2. Peng Y, Hu Z, Wen M, et al. Development and validation of a cranial ultrasound imaging-based deep learning model for periventricular-intraventricular haemorrhage detection and grading: a two-centre study. Pediatr Radiol. 2025;55:2076-2085. [doi:10.1007/s00247-025-06327-x](https://doi.org/10.1007/s00247-025-06327-x).
3. Kim KY, Nowrangi R, McGehee A, Joshi N, Acharya PT. Assessment of germinal matrix hemorrhage on head ultrasound with deep learning algorithms. Pediatr Radiol. 2022;52:533-538. [doi:10.1007/s00247-021-05239-w](https://doi.org/10.1007/s00247-021-05239-w).
4. Ibrahim NM, Alanize H, Alqahtani L, et al. Deep Learning Approaches for the Assessment of Germinal Matrix Hemorrhage Using Neonatal Head Ultrasound. Sensors. 2024;24:7052. [doi:10.3390/s24217052](https://doi.org/10.3390/s24217052).
5. Tabrizi PR, Obeid R, Cerrolaza JJ, et al. Automatic Segmentation of Neonatal Ventricles from Cranial Ultrasound for Prediction of Intraventricular Hemorrhage Outcome. EMBC. 2018:3136-3139. [doi:10.1109/EMBC.2018.8513097](https://doi.org/10.1109/EMBC.2018.8513097).
6. Benavente-Fernandez I, et al. Automatic segmentation of ventricular volume by 3D ultrasonography in post haemorrhagic ventricular dilatation among preterm infants. Sci Rep. 2021;11. [doi:10.1038/s41598-020-80783-3](https://doi.org/10.1038/s41598-020-80783-3).
7. Szentimrey Z, de Ribaupierre S, Fenster A, Ukwatta E. Semi-supervised learning framework with shape encoding for neonatal ventricular segmentation from 3D ultrasound. Med Phys. 2024;51. [doi:10.1002/mp.17242](https://doi.org/10.1002/mp.17242).
8. Jiao J, Zhou J, Li X, et al. USFM: A Universal Ultrasound Foundation Model Generalized to Tasks and Organs towards Label Efficient Image Analysis. 2024. [arXiv:2401.00153](https://arxiv.org/abs/2401.00153).
9. Pham HN, et al. Validity of ChatGPT in Assisting Diagnosis of Periventricular-Intraventricular Hemorrhage via Cranial Ultrasound Imaging in Very Preterm Infants. 2025. [PubMed 40376373](https://pubmed.ncbi.nlm.nih.gov/40376373/).

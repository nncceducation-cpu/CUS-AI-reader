# Canadian consensus clinical specification

This specification implements the 2021 Canadian consensus paper and cross-checks the branch sequence against the HUS Diagnostic application. It is a software specification for research use, not a substitute for the paper or a complete diagnostic reporting standard.

## Unit of classification

Classify left and right cerebral hemispheres independently. Do not collapse bilateral injury to the worst side in the stored record. A study-level summary can identify the worst finding after both sides have been preserved.

## Required evidence

A complete examination requires multiple coronal and sagittal or parasagittal anterior-fontanel views plus posterior fossa assessment through the mastoid window. Posterior fontanel views can help distinguish occipital horn blood from artifact. A finding seen in only one plane should be treated cautiously.

For cine input, every decodable frame must be evaluated. A display preview can show fewer frames, but it cannot determine which frames enter plane detection or lesion aggregation. Coronal and sagittal or parasagittal coverage are recorded separately and require clinician confirmation before a whole-study classification can be marked final.

The image showing the greatest GMH-IVH severity in the first postnatal week is used for initial grading. Serial imaging is needed for WMI evolution, PVHI cavitation, and PHVD trajectory.

## GMH-IVH and PVHI rules

### Step 1

Is hemorrhage present within or around the germinal matrix or within the lateral, third, or fourth ventricle?

- No: negative for GMH-IVH, then assess periventricular white matter.
- Yes: proceed to Step 2.

### Step 2

Is the hemorrhage confined to the germinal matrix?

- Yes, with no blood in the lateral ventricle: Grade I GMH-IVH.
- No, with blood in the lateral ventricle or on the choroid plexus: proceed to Step 3.

### Step 3

Does the intraventricular hemorrhage acutely distend the ipsilateral lateral ventricle and is AHW above 6 mm?

- Both criteria yes: Grade III GMH-IVH.
- Either criterion no: Grade II GMH-IVH.
- Missing distension or AHW evidence: indeterminate Grade II versus III.

The threshold is strictly above 6 mm. A recorded value of 6.0 mm is not above the threshold.

### Step 4

Is focal periventricular white-matter echogenicity present adjacent to the side of GMH-IVH, or adjacent to the largest hemorrhage in bilateral disease?

- With ipsilateral Grade I, II, or III GMH-IVH: report PVHI in addition to the GMH-IVH grade.
- Without GMH-IVH: PVHI is not the correct label. Assess for ischemic WMI and other causes.

## White matter injury

Abnormal periventricular echogenicity is brighter than the choroid plexus or inhomogeneous. Subtle homogeneous echogenicity that does not exceed the choroid plexus can be physiologic. Serial evolution defines the grade:

- Grade 1: transient increased PVE persisting for 7 days or longer, without cystic evolution.
- Grade 2: PVE evolves into small localized frontoparietal periventricular cysts.
- Grade 3: PVE evolves into extensive fronto-parieto-occipital periventricular cysts.
- Grade 4: PVE evolves into extensive cysts in deep or subcortical white matter.

Ischemic WMI is more often bilateral. PVHI is commonly asymmetric and ipsilateral to GMH-IVH. Serial studies help distinguish unilateral cystic evolution of PVHI from ischemic WMI.

## Cerebellar hemorrhage

Use the mastoid window and assess in two planes when possible:

- Punctate: 4 mm or smaller, often seen on MRI and not readily seen on cUS.
- Limited: above 4 mm and below one third of a cerebellar hemisphere.
- Large: one third or more of a cerebellar hemisphere.

## Ventricular measurement and PHVD

Measure VI and AHW on the coronal plane at the level of the foramen of Monro. Measure left and right separately. When both attempts are technically valid, use the larger measurement for severity.

Grade III GMH-IVH refers to acute clot-related ventricular distension in the first week. PHVD usually appears 7 to 10 days after the hemorrhage and can develop up to 2 to 3 weeks later.

- Moderate PHVD: VI above the 97th centile and AHW above 6 mm.
- Severe PHVD: VI above the 97th centile plus 4 mm, or AHW above 10 mm.

Ventricular enlargement without preceding hemorrhage is not PHVD. Consider ex-vacuo dilatation, obstruction, or another cause.

## Severe preterm brain injury flag

The consensus paper proposes a quality-improvement definition based on any of these findings:

- Grade III IVH or PVHI
- severe ventricular dilatation
- cystic white matter lesions
- moderate to large cerebellar hemorrhage

The current software uses a conservative flag for Grade III GMH-IVH, PVHI, severe PHVD, Grade 2 to 4 WMI, or large CBH. The phrase in the source paper includes moderate CBH, but its size classification lists punctate, limited, and large. The operational mapping of limited versus moderate CBH requires author decision before a research protocol is locked.

## Conditions that force abstention

- Missing required planes
- Only one still image for a study-level normal classification
- Unknown laterality
- Hemorrhage location or intraventricular extension not established
- Grade II versus III attempted without acute distension and AHW evidence
- WMI grade attempted without duration or serial cystic evolution
- PHVD attempted without prior hemorrhage and serial VI or AHW measurements
- Measurement attempted without validated scale
- Clinician has not verified model-suggested features

## Source

Mohammad K, Scott JN, Leijser LM, et al. Front Pediatr. 2021;9:618236. [doi:10.3389/fped.2021.618236](https://doi.org/10.3389/fped.2021.618236).

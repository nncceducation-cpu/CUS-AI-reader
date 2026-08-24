# AI scoring accuracy, version 0.6.0

This note records what was wrong with the scoring path, what replaced it, and
what evidence supports each change. It is written so that a reviewer can check
the reasoning without reading the diff.

## Summary of the defects

| # | Defect | Effect on accuracy |
|---|--------|--------------------|
| 1 | Frame to study pooling was the mean of the three highest frame probabilities | On a 400-frame sweep this is close to a maximum. Three noisy frames were enough to declare any finding present. |
| 2 | Evidence fields were resolved by taking the highest-scoring alias | One-sided error. `hemorrhage_present` listed `intraventricular_blood` as an alias, so intraventricular scores could raise hemorrhage presence but never lower it. |
| 3 | `confined_to_germinal_matrix` had no label in the manifest contract | Step 2 was unanswerable, so every AI grade was pinned at "Indeterminate GMH-IVH grade". |
| 4 | A label literally named `pvhi` was accepted as the input for periventricular echogenicity | Category error. PVHI is the conclusion Step 4 reaches, so feeding it back as an input short-circuited the rule. |
| 5 | Abstention was global | Any one uncertain feature, plus the default absence of a serial study, suppressed every domain. In practice the tool abstained on every study. |
| 6 | Probabilities were thresholded raw | Multi-task heads are over-confident. A threshold of 0.50 on uncalibrated output is not the operating point its number implies. |
| 7 | No constraint linked related heads | The model could report P(AHW > 10 mm) = 0.80 with P(AHW > 6 mm) = 0.40, which is anatomically impossible, and the rules would grade the contradiction. |
| 8 | Every label was scored from every accepted frame | Cerebellar labels were scored from coronal frontal-horn frames, where the cerebellum is not in view. |

## What replaced them

### Persistence aggregation, `cus_ai/aggregation.py`

A lesion persists. As the probe sweeps through a germinal matrix clot the clot
appears on a contiguous run of frames and reappears in the orthogonal plane.
Speckle, the choroid plexus edge, and the caudothalamic notch produce isolated
bright frames instead. The score is now the more conservative of two statistics:

- the k-th largest calibrated probability, where k is the number of frames the
  finding must be visible on, so the score is the level that at least k frames
  reach
- the highest level sustained across a run of consecutive frames

Frames are weighted by plane confidence and image quality. Labels are restricted
to the planes that show the relevant anatomy. Where a finding can be seen in two
planes, single-plane findings are held back, which follows the consensus caution
that echogenicity seen in only one plane should be read as artifact.

Studies exported as single still images relax the run requirement to the number
of frames supplied, so still-image centres remain gradeable.

### Derived evidence, `cus_ai/evidence_mapping.py`

Alias lookup is gone. Fields a network can be trained to emit are read directly.
Fields that are logical consequences are derived, exactly as the consensus
algorithm derives them. Step 2 does not need its own detector: hemorrhage
present with no intraventricular blood is confined to the germinal matrix. Step 1
takes the union of the germinal matrix and intraventricular heads, because that
is the question the step asks.

### Per-domain abstention, `cus_ai/ai_consensus.py`

Each reportable domain carries its own status, confidence, and reasons. A study
with a clean coronal sweep and no mastoid window now reports its GMH-IVH grades
and withholds cerebellar hemorrhage, which is what a reader does. The study-level
abstention flag fires only when no domain survives. WMI grade and PHVD remain
withheld without a serial study, because both are defined by change over time.

### Calibration, `cus_ai/calibration.py`

Platt scaling, isotonic interpolation, or a global temperature, fitted offline
and carried in the manifest. `scripts/fit_operating_point.py` fits calibration
and selects thresholds from labelled studies, weighting a false negative above a
false positive by default, since a missed grade III bleed and an unnecessary
second read are not comparable errors.

### Anatomic constraints, `cus_ai/consistency.py`

Nested thresholds are made monotone, containment hierarchies are enforced, and
mutually exclusive grade families are renormalised. Every repair is recorded.
Frequent repair is itself a finding: it means the model's calibration or its
training targets need attention.

## Evidence

### Simulation

`scripts/benchmark_aggregation.py` isolates the pooling question, which can be
answered without labelled images. Sweeps are generated with a contiguous lesion
run in positive studies, artifact arriving in short bursts in both, per-study
variation in how visible the lesion is, and a share of positives carrying a
subtle lesion. Rules are compared at matched sensitivity so the result is not a
threshold shift.

At the default setting, 1500 studies, 30 percent prevalence, artifact rate 0.005
per frame in bursts up to 3 frames:

| Rule | AUC | Specificity at 90 percent sensitivity |
|------|-----|----------------------------------------|
| Legacy top-k mean | 0.78 | 0.47 |
| Persistence | 0.92 | 0.82 |

Run `--sweep` for the full range of artifact rates. Persistence wins across all
of them, by a wider margin when artifact is isolated and a narrower one when
artifact arrives in long bursts, which is the expected behaviour.

This is a simulation of the pooling step. It is not a clinical validation and it
says nothing about how well any model reads an ultrasound.

### Calibration

On a synthetic over-confident detector, Platt scaling moved log loss from 0.328
to 0.096 and the Brier score from 0.097 to 0.018, and removed the false
positives that a naive 0.50 threshold produced. Reproduce with the worked
example in `scripts/fit_operating_point.py`.

### Rule engine

`tests/test_consensus_rules.py` pins each clinical rule to the paper, including
the choroid brightness definition of echodensity, the ischemic-injury branch of
Step 4, the bilateral Step 4 anchor, the day 7 PHVD boundary, and the inclusion
of moderate cerebellar hemorrhage in the severe-injury definition.

## What is still missing

No trained weights ship with this repository, and none should until a model has
been fitted and externally validated on adjudicated Canadian data. Everything
above improves how model output is turned into a grade. None of it substitutes
for the model. The manifest stays `"validated": false` until held-out
performance has been measured and reported.

## Source

Mohammad K, Scott JN, Leijser LM, et al. Consensus Approach for Standardizing the
Screening and Classification of Preterm Brain Injury Diagnosed With Cranial
Ultrasound: A Canadian Perspective. Front Pediatr. 2021;9:618236.
doi:10.3389/fped.2021.618236

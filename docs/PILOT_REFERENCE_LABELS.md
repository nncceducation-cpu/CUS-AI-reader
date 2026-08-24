# Pilot reference labels, version 1.0.0

## Status

The project lead supplied one diagnostic summary for each of the 15 pilot examinations. These records are stored as a provisional single-reader label set. They are suitable for workflow testing and for planning a formal annotation study. They are not an adjudicated reference standard and cannot support diagnostic performance claims.

The app conceals the matched pilot label until an independent expert form has been submitted. Agreement calculations include only domains explicitly supplied by the project lead. An unmentioned domain is `not_reported`, not negative.

## Normalization rules

- Grade I, II, and III are stored separately for the left and right hemispheres.
- PVHI is stored separately from the GMH-IVH grade.
- A porencephalic cyst is stored as the ipsilateral cystic evolution of previous PVHI. It is not converted to ischemic cystic WMI.
- CPVL grade 3 is normalized to grade 3 ischemic WMI.
- Severe PHVD is retained as supplied. VI and AHW threshold fields remain unavailable until measurements are entered.
- A study labelled normal is treated as negative across GMH-IVH, PVHI, porencephalic cyst, ischemic WMI, CBH, and PHVD.
- If laterality or a required precursor grade was not supplied, the field is `indeterminate` or `not_reported` and is not silently completed.

## Records requiring clarification or adjudication

| Study | Issue |
|---|---|
| Case 8 day 5 | The left PVHI implies ipsilateral GMH-IVH, but the left GMH-IVH grade was not supplied. |
| Case 7 | Examination age was not supplied. |
| Case 4 day 5 | Bilateral PVHI implies bilateral GMH-IVH, but the grades were not supplied. |
| Case 4 day 21 | Severe PHVD was supplied without VI or AHW measurements. |
| Case 1 day 15 | Severe PHVD was supplied without VI or AHW measurements. Prior bilateral grade III GMH-IVH is linked from day 5. |

## Required next annotation step

Two qualified readers should score each examination independently while blinded to these labels and to AI output. Initial readings must be saved before adjudication. Disagreements should be resolved by a third qualified reader or a consensus panel. Model development must split data by infant, not frame or examination, and requires a substantially larger cohort than these eight infants.

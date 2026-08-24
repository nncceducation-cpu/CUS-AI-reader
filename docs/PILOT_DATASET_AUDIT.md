# Pilot dataset audit

## Scope received

The deidentified shared folder contains eight infant codes and 15 identifiable examination time points. Most infants have two serial examinations. The folder includes 15 compressed whole-examination AVI files, 28 plane-specific AVI files, and three ventricular-measurement still images. One late examination has its coronal and sagittal clips stored in separate folders with an ambiguous folder name and should be reconciled before annotation.

No expert grading spreadsheet, radiology report, frame annotation, lesion mask, scanner inventory, gestational age table, or center variable was present at audit. The media must therefore be treated as an unlabeled pilot dataset.

## Technical verification

A representative combined AVI file was downloaded without altering the source. The container reported 504 frames and all 504 frames decoded sequentially. Resolution was 1280 by 720 pixels at 29.97 frames per second. Visual review confirmed both coronal and sagittal sweeps in the combined clip. This test supports compatibility with the current exhaustive frame decoder, but it does not establish compatibility for every file.

## Permitted use in this project

This dataset can support:

- media-ingestion and exhaustive-frame testing
- expert annotation workflow development
- serial-study data-linkage testing
- interface usability testing
- creation of an adjudicated pilot reference set

It cannot support a defensible model-training split, independent validation, or performance claims. Frames from the same infant are highly correlated and must never be divided between training and evaluation partitions.

## Annotation plan

Each examination should be graded independently by two qualified readers who are blinded to AI output. Store the initial labels before adjudication. A third expert should resolve disagreements. The minimum reference record includes infant code, examination code, age at scan, gestational age, coronal and sagittal adequacy, posterior fossa adequacy, left and right consensus features, AHW and VI measurements when scale is valid, WMI evolution, CBH, PHVD, and the final Canadian consensus categories.

The app version 0.5.0 exports expert inputs, AI probabilities, every-frame plane assignments, AI consensus results, exact agreement, and abstention reasons as CSV. It also stores the 15 project-lead labels as a provisional single-reader reference set. These records form the annotation table for later model development.

## Data-governance actions

- Change the shared-folder role from link-access writer to link-access viewer after uploads are complete.
- Keep the source media outside the public GitHub repository.
- Confirm that the approved secondary-use scope includes AI model development and external expert review.
- Use an encrypted institutional location for the working copy and the codebook.
- Do not upload exported records that contain dates or free text until they have passed a re-identification review.

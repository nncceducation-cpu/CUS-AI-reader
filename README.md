# CUS AI Reader

CUS AI Reader is a research-use web application for neonatal cranial ultrasound study review. It accepts a single image, an image set, DICOM objects, or cine clips. The app decodes every frame sequentially, performs technical quality checks on every frame, and sends every frame to an installed validated model. The model contract identifies coronal, sagittal, posterior fossa, other, or indeterminate planes before plane-specific feature aggregation. Clinician-verified findings then enter the Canadian consensus GMH-IVH and PVHI rules independently for the left and right hemispheres. The app also classifies serial WMI, cerebellar hemorrhage, and PHVD, then exports an auditable JSON or Markdown record.

The repository also provides a guarded ONNX interface for a future validated feature model. No diagnostic weights are bundled. The software will not silently substitute heuristics or a general-purpose vision-language model for a validated neonatal ultrasound model.

## Safety status

This is not a medical device and does not create a consultative diagnostic ultrasound report. It is intended for research, model development, education, and quality improvement. It must not be used for treatment decisions or family counselling. Use de-identified media only. A qualified neonatologist, pediatric radiologist, or neuroradiologist must review the complete examination.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the displayed local URL, accept the research-use gate, then:

1. Upload images, DICOM files, or a clip in the Media tab.
2. Confirm that every source frame decoded, then review the display preview and complete technical QC table.
3. Record verified findings in the Evidence tab.
4. Review and export the consensus classification in the Report tab.

## Run with Docker

```powershell
docker build -t cus-ai-reader .
docker run --rm -p 8501:8501 cus-ai-reader
```

Then open `http://localhost:8501`.

## Install a validated ONNX feature model

1. Copy the ONNX file into `models/`.
2. Copy and edit `models/manifest.example.json` as `models/<model-id>.manifest.json`.
3. Set `validated` to `true` only after the validation protocol has been completed and approved.
4. Install `requirements-ml.txt`.
5. Restart the app.

The current contract expects an ONNX output shaped `[batch, labels]`, with one probability per manifest label. `plane_coronal` and `plane_sagittal` outputs are mandatory. The reader processes all frames in sequential batches, retains every per-frame result, and aggregates feature probabilities within accepted planes. Model probabilities remain separate from clinician-verified evidence. The deterministic consensus rules are in `cus_ai/clinical.py`.

## Repository map

- `app.py`: Streamlit interface
- `cus_ai/media.py`: image, DICOM, and clip ingestion
- `cus_ai/model.py`: guarded ONNX adapter and abstention state
- `cus_ai/clinical.py`: deterministic Canadian consensus rule engine
- `cus_ai/reporting.py`: auditable report export
- `docs/ALGORITHM_REVIEW.md`: evidence review and selected architecture
- `docs/CLINICAL_SPEC.md`: clinical classification specification
- `docs/VALIDATION_PROTOCOL.md`: development and external validation protocol
- `docs/DATA_DICTIONARY.md`: annotation and inference schema
- `docs/SECURITY_PRIVACY.md`: privacy and deployment controls
- `tests/`: executable rule, media, model, and reporting checks

## Clinical source

Mohammad K, Scott JN, Leijser LM, et al. Consensus Approach for Standardizing the Screening and Classification of Preterm Brain Injury Diagnosed With Cranial Ultrasound: A Canadian Perspective. Front Pediatr. 2021;9:618236. [doi:10.3389/fped.2021.618236](https://doi.org/10.3389/fped.2021.618236).

The HUS Diagnostic application was used to cross-check the four-step branching and report wording: [nncceducation-cpu/HUSDiagnostic](https://github.com/nncceducation-cpu/HUSDiagnostic).

## Current status

- Media ingestion: implemented
- Single image, image set, DICOM, and exhaustive cine-frame decoding: implemented
- Coronal and sagittal per-frame model contract and plane-aware aggregation: implemented
- Technical QC: implemented
- Consensus rule engine: implemented and tested
- Structured export: implemented
- Diagnostic model weights: not available and not represented as complete
- External clinical validation: not started
- Regulatory authorization: not started

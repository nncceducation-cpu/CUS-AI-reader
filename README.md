# CUS AI Reader

CUS AI Reader is a research-use web application for neonatal cranial ultrasound study review. It accepts a single image, an image set, DICOM objects, or cine clips. The app decodes every frame sequentially, performs technical quality checks on every frame, and sends every frame to an installed model. The model contract identifies coronal, sagittal, posterior fossa, other, or indeterminate planes before plane-specific feature aggregation. AI feature decisions and independently entered expert findings enter the same Canadian consensus rule engine. The app displays AI grading, expert grading, exact agreement, and auditable JSON, Markdown, study-level CSV, and every-frame CSV exports.

Version 0.6.0 rebuilds the scoring path between the model and the rule engine: calibrated frame probabilities, persistence-based frame-to-study aggregation, anatomic consistency constraints, derived rather than alias-matched consensus evidence, and abstention scoped to the domain it affects instead of the whole study. The reasoning, the defects it corrects, and the supporting simulation are in `docs/AI_SCORING_ACCURACY.md`.

The repository also provides a guarded ONNX interface for a future validated feature model. No diagnostic weights are bundled. The software will not silently substitute heuristics or a general-purpose vision-language model for a validated neonatal ultrasound model. Version 0.5.0 includes a provisional single-reader label registry for the 15 pilot examinations. The supplied labels remain concealed until an independent expert score has been submitted.

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
3. Run the installed model in the AI grading tab.
4. Record an independent expert grade without accepting AI suggestions in the Expert grading tab.
5. Review agreement and export study-level and every-frame raw CSV records.
6. In the agreement tab, reveal a matched pilot reference label only after completing the independent read.

## Portable offline Windows edition

The portable Windows package includes an isolated Python runtime and locked application dependencies. It does not require a separate Python installation or an internet connection after the ZIP has been created.

1. Move the versioned offline ZIP to `C:\` or another short location before extraction.
2. Extract the complete ZIP without skipping any file. It creates a short `CUSAI` folder to avoid Windows path-length failures.
3. Open `CUSAI` and double-click `Start CUS AI Reader.cmd`.
4. Use the browser page opened at `127.0.0.1`.
5. Close the launcher window to stop the local server.

If the browser does not open automatically, keep the launcher window open and double-click `CUS AI Reader Local Page.url`. Startup details are normally written to `%LOCALAPPDATA%\CUS-AI-reader\startup.log`, with the Windows temporary folder used as a fallback.

If Windows reports `Path too long`, cancel extraction and use a shorter destination. Do not select `Skip`, because an incomplete pandas or compiled-extension installation can crash the technical-quality table. The launcher checks the runtime before starting and reports an incomplete extraction clearly.

The launcher binds Streamlit to the loopback interface only, disables Streamlit usage telemetry, and prevents the file watcher from starting. Uploaded media remains in the active local process. Reports are written only when the user downloads them.

Maintainers can reproduce the portable ZIP on Windows with Python 3.12 by running `scripts/build_portable_windows.ps1`. The build uses the official Python 3.12.10 embedded distribution, a locked dependency file, a full launcher startup check, and SHA256 manifests.

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
- `cus_ai/ai_consensus.py`: probability-to-feature decisions and AI consensus grading
- `cus_ai/agreement.py`: expert versus AI agreement and raw CSV exports
- `cus_ai/reference_labels.py`: validated pilot-label loading, study matching, and reference agreement
- `cus_ai/clinical.py`: deterministic Canadian consensus rule engine
- `cus_ai/reporting.py`: auditable report export
- `docs/ALGORITHM_REVIEW.md`: evidence review and selected architecture
- `docs/CLINICAL_SPEC.md`: clinical classification specification
- `docs/VALIDATION_PROTOCOL.md`: development and external validation protocol
- `docs/PILOT_DATASET_AUDIT.md`: audit and permitted role of the eight-infant pilot dataset
- `docs/PILOT_REFERENCE_LABELS.md`: normalization decisions for the 15 supplied labels
- `docs/AI_CONSENSUS_METHOD.md`: feature targets, abstention, and model-development method
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
- Separate AI grading and independent expert grading: implemented
- Expert versus AI exact agreement: implemented
- Provisional reference-label matching and expert or AI versus reference agreement: implemented
- Porencephalic cyst as an evolved PVHI target, separate from ischemic WMI: implemented
- Study-level and every-frame raw CSV export: implemented
- Structured export: implemented
- Portable offline Windows x64 launcher and reproducible bundle: implemented
- Diagnostic model weights: not available and not represented as complete
- External clinical validation: not started
- Regulatory authorization: not started

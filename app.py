from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import streamlit as st

from cus_ai.agreement import agreement_summary, compare_classifications, frame_csv, study_csv
from cus_ai.ai_consensus import grade_prediction
from cus_ai.clinical import classify_study
from cus_ai.media import MediaFrame, decode_media, quality_metrics
from cus_ai.model import OnnxFeatureModel, discover_model, prediction_to_json
from cus_ai.reference_labels import (
    compare_to_reference,
    find_reference_label,
    load_reference_labels,
    reference_csv,
    reference_display_row,
)
from cus_ai.reporting import build_report, report_to_markdown
from cus_ai.schemas import SideEvidence, StudyEvidence


APP_VERSION = "0.5.0"
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
REFERENCE_LABEL_PATH = ROOT / "data" / "pilot_reference_labels_v1.csv"
OFFLINE_MODE = os.environ.get("CUS_AI_OFFLINE", "0") == "1"


st.set_page_config(
    page_title="CUS AI Reader",
    page_icon="brain",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f6f8fb; }
      .block-container { max-width: 1260px; padding-top: 2rem; }
      h1, h2, h3 { color: #102a43; letter-spacing: -0.02em; }
      [data-testid="stMetric"] { background: white; border: 1px solid #d9e2ec; border-radius: 12px; padding: 12px; }
      .research-banner { background: #fff8e6; border: 1px solid #f5c66d; border-radius: 12px; padding: 14px 16px; color: #6b4e16; }
      .result-card { background: white; border: 1px solid #d9e2ec; border-radius: 14px; padding: 16px; margin-bottom: 10px; }
      .tiny { color: #627d98; font-size: 0.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, max_entries=2)
def cached_decode(name: str, payload: bytes):
    return decode_media(name, payload)


@st.cache_data(show_spinner=False)
def cached_reference_labels() -> list[dict[str, str]]:
    return load_reference_labels(REFERENCE_LABEL_PATH)


def preview_indices(total: int, limit: int) -> list[int]:
    """Select display thumbnails only. Processing always uses every decoded frame."""
    if total <= limit:
        return list(range(total))
    if limit == 1:
        return [0]
    return sorted({round(index * (total - 1) / (limit - 1)) for index in range(limit)})


def answer_select(label: str, key: str, help_text: str | None = None) -> str:
    options = {
        "Unknown or not assessed": "unknown",
        "Yes": "yes",
        "No": "no",
    }
    selected = st.selectbox(label, list(options), key=key, help=help_text)
    return options[selected]


def side_form(side: str) -> SideEvidence:
    title = side.capitalize()
    st.markdown(f"### {title} hemisphere")
    hemorrhage = answer_select(
        "Hemorrhage within or around germinal matrix or ventricular system",
        f"{side}_hemorrhage",
    )
    confined = answer_select("Hemorrhage confined to germinal matrix", f"{side}_confined")
    intraventricular = answer_select("Blood in lateral ventricle or on choroid plexus", f"{side}_ivh")
    distension = answer_select("Acute ipsilateral lateral ventricular distension", f"{side}_distension")
    measure_ahw = st.checkbox("AHW was measured on an appropriate coronal plane", key=f"{side}_has_ahw")
    ahw = None
    if measure_ahw:
        ahw = st.number_input(
            "Anterior horn width, mm",
            min_value=0.0,
            max_value=40.0,
            value=4.0,
            step=0.1,
            key=f"{side}_ahw",
        )
    pve = answer_select(
        "Focal periventricular echogenicity adjacent to ipsilateral GMH-IVH",
        f"{side}_pve",
    )
    brighter = answer_select(
        "Abnormal white matter echogenicity is brighter than choroid plexus",
        f"{side}_brighter",
    )
    cystic_options = {
        "Not assessed": "not_assessed",
        "No porencephalic cyst": "none",
        "Porencephalic cyst consistent with evolved PVHI": "porencephalic",
        "Multiple unilateral cysts consistent with evolved PVHI": "multiple_evolved_pvhi",
    }
    cystic_label = st.selectbox("Cystic sequela on this side", list(cystic_options), key=f"{side}_cystic")
    verified = st.checkbox("Clinician verified the recorded features", key=f"{side}_verified")
    return SideEvidence(
        side=side,  # type: ignore[arg-type]
        hemorrhage_present=hemorrhage,  # type: ignore[arg-type]
        confined_to_germinal_matrix=confined,  # type: ignore[arg-type]
        intraventricular_blood=intraventricular,  # type: ignore[arg-type]
        ventricular_distension=distension,  # type: ignore[arg-type]
        ahw_mm=float(ahw) if ahw is not None else None,
        ahw_above_6_mm=("yes" if ahw > 6 else "no") if ahw is not None else "unknown",
        ahw_above_10_mm=("yes" if ahw > 10 else "no") if ahw is not None else "unknown",
        adjacent_periventricular_echogenicity=pve,  # type: ignore[arg-type]
        echogenicity_brighter_than_choroid=brighter,  # type: ignore[arg-type]
        cystic_change=cystic_options[cystic_label],
        clinician_verified=verified,
    )


def disclaimer_gate() -> None:
    if st.session_state.get("disclaimer_accepted"):
        return
    st.title("CUS AI Reader")
    st.caption("Neonatal cranial ultrasound research prototype")
    st.markdown(
        """
        <div class="research-banner">
        This software is for research, model development, education, and quality improvement only.
        It is not a medical device, does not generate a consultative diagnostic ultrasound report,
        and must not be used to make treatment or counselling decisions. A qualified neonatologist,
        pediatric radiologist, or neuroradiologist must review the complete examination.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    accepted = st.checkbox("I understand the intended use and will not use this prototype for clinical care.")
    if st.button("Enter research workspace", type="primary", disabled=not accepted):
        st.session_state.disclaimer_accepted = True
        st.rerun()
    st.stop()


def render_media_tab() -> tuple[list[MediaFrame], list[dict], list[str]]:
    st.subheader("1. Load a study")
    st.write(
        "Upload one image, an image set, a DICOM object, or a cine clip. Processing stays in this app session. "
        "Use de-identified source files only."
    )
    st.success("Every decodable frame is processed in sequence. The preview limit below affects display only.")
    preview_limit = st.slider("Maximum frame thumbnails shown", 8, 64, 24, 8)
    uploads = st.file_uploader(
        "Cranial ultrasound media",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "gif", "dcm", "dicom", "mp4", "mov", "avi", "mkv", "webm", "m4v"],
        accept_multiple_files=True,
    )
    all_frames: list[MediaFrame] = []
    warnings: list[str] = []
    media_summary: list[dict] = []
    study_hasher = hashlib.sha256()
    for upload in uploads or []:
        try:
            payload = upload.getvalue()
            study_hasher.update(upload.name.encode("utf-8"))
            study_hasher.update(payload)
            result = cached_decode(upload.name, payload)
            all_frames.extend(result.frames)
            warnings.extend(f"{upload.name}: {item}" for item in result.warnings)
            media_summary.append(
                {
                    "name": upload.name,
                    "frames_loaded": len(result.frames),
                    "source_frame_count": result.technical_metadata.get("source_frame_count", len(result.frames)),
                    "all_frames_processed": result.technical_metadata.get("all_frames_processed", True),
                    "technical_metadata": result.technical_metadata,
                }
            )
        except Exception as exc:  # user-facing boundary
            warnings.append(f"{upload.name}: {exc}")

    media_fingerprint = study_hasher.hexdigest() if uploads else None
    if st.session_state.get("media_fingerprint") != media_fingerprint:
        st.session_state.media_fingerprint = media_fingerprint
        st.session_state.pop("model_prediction", None)
        st.session_state.pop("ai_consensus_result", None)
        st.session_state.pop("study_evidence", None)
        st.session_state.pop("study_classification", None)

    if warnings:
        for warning in warnings:
            st.warning(warning)
    if not all_frames:
        st.info("No readable study has been loaded yet.")
        return [], media_summary, warnings

    reviewable = 0
    metrics_rows = []
    for frame in all_frames:
        metrics = quality_metrics(frame.image)
        reviewable += metrics["quality_flag"] == "reviewable"
        metrics_rows.append(
            {
                "source": frame.source_name,
                "frame": frame.frame_index,
                **metrics,
                "pixel spacing": frame.pixel_spacing_mm or "not available",
            }
        )
    complete_files = sum(bool(item.get("all_frames_processed")) for item in media_summary)
    a, b, c, d = st.columns(4)
    a.metric("Files", len(media_summary))
    b.metric("Frames processed", len(all_frames))
    c.metric("Reviewable by basic QC", f"{reviewable}/{len(all_frames)}")
    d.metric("Complete decodes", f"{complete_files}/{len(media_summary)}")

    st.markdown("### Frame review")
    st.caption(
        "All frames contribute to QC and installed-model inference. Thumbnails are an evenly spaced display preview only. "
        "QC flags do not establish the anatomic plane or diagnostic adequacy."
    )
    shown = set(preview_indices(len(all_frames), preview_limit))
    columns = st.columns(4)
    display_index = 0
    for index, (frame, row) in enumerate(zip(all_frames, metrics_rows)):
        if index not in shown:
            continue
        with columns[display_index % 4]:
            st.image(frame.image, width="stretch")
            st.caption(f"{frame.source_name} | frame {frame.frame_index} | {row['quality_flag']}")
        display_index += 1
    with st.expander("Technical quality table"):
        st.dataframe(metrics_rows, width="stretch", hide_index=True)
    return all_frames, media_summary, warnings


def render_model_panel(frames: list[MediaFrame], media_summary: list[dict]) -> None:
    st.subheader("2. AI grading")
    st.write(
        "The installed model evaluates every decoded frame, assigns the imaging plane, estimates consensus features, "
        "then sends those feature decisions to the same deterministic Canadian consensus rule engine used for expert grading."
    )
    manifest, model_warnings = discover_model(MODEL_DIR)
    for warning in model_warnings:
        st.info(warning)
    if manifest is None:
        st.warning(
            "Automatic image grading is unavailable because no trained neonatal cranial-ultrasound model is installed. "
            "The clip reader and expert-entered Canadian consensus classifier remain available below."
        )
        st.caption(
            "The app will not substitute generic image heuristics for a validated model. Review the complete examination, "
            "record the verified findings, then select Classify with consensus rules."
        )
        return
    st.write(f"Model: `{manifest.model_id}` version `{manifest.version}`")
    st.write(f"Intended use: {manifest.intended_use}")
    if not frames:
        st.info("Load media before running the installed model.")
        return
    model_exists = (MODEL_DIR / manifest.onnx_file).exists()
    research_ack = True
    if model_warnings and model_exists:
        research_ack = st.checkbox(
            "Run this guarded model for research testing despite the warnings above",
            value=False,
        )
    can_run = model_exists and research_ack
    if st.button("Analyze every frame with installed model", disabled=not can_run):
        with st.spinner(f"Analyzing all {len(frames)} frames in order..."):
            try:
                model = OnnxFeatureModel(MODEL_DIR, manifest)
                prediction = model.predict(frames)
                prediction_json = prediction_to_json(prediction)
                st.session_state.model_prediction = prediction_json
                all_frames_processed = bool(media_summary) and all(
                    bool(item.get("all_frames_processed")) for item in media_summary
                )
                ai_result = grade_prediction(
                    prediction_json,
                    thresholds=manifest.thresholds,
                    decision_margin=manifest.decision_margin,
                    all_frames_processed=all_frames_processed,
                )
                st.session_state.ai_consensus_result = ai_result.to_dict()
            except Exception as exc:
                st.error(f"Inference failed safely: {exc}")
    if prediction := st.session_state.get("model_prediction"):
        if prediction["abstained"]:
            st.warning("Model abstained: " + "; ".join(prediction["abstention_reasons"]))
        counts = prediction["plane_counts"]
        a, b, c, d = st.columns(4)
        a.metric("Frames analyzed", prediction["processed_frame_count"])
        b.metric("Accepted coronal", counts.get("coronal", 0))
        c.metric("Accepted sagittal", counts.get("sagittal", 0))
        d.metric("Indeterminate plane", counts.get("indeterminate", 0))
        frame_rows = [
            {
                "source": row["source_name"],
                "frame": row["frame_index"],
                "plane": row["plane"],
                "plane confidence": round(row["plane_confidence"], 4),
                "ambiguous": row["ambiguous_plane"],
            }
            for row in prediction["frame_predictions"]
        ]
        with st.expander("Per-frame plane assignments"):
            st.dataframe(frame_rows, width="stretch", hide_index=True)
        with st.expander("Study-level model probabilities"):
            st.json(
                {
                    "probabilities": prediction["probabilities"],
                    "probabilities_by_plane": prediction["probabilities_by_plane"],
                }
            )
    if ai_result := st.session_state.get("ai_consensus_result"):
        st.markdown("### AI Canadian consensus grade")
        if ai_result["abstained"]:
            st.warning("AI abstained from a final grade: " + "; ".join(ai_result["abstention_reasons"]))
        else:
            st.success("AI produced an accepted research grade for expert comparison.")
        result = ai_result["classification"]
        left_result, right_result = st.columns(2)
        with left_result:
            render_side_result("AI left hemisphere", result["left"])
        with right_result:
            render_side_result("AI right hemisphere", result["right"])
        st.write(f"**AI white matter injury:** {result['wmi']}")
        st.write(f"**AI cerebellar hemorrhage:** {result['cerebellar_hemorrhage']}")
        st.write(f"**AI PHVD:** {result['phvd']}")
        with st.expander("AI feature decisions and missing outputs"):
            st.json(
                {
                    "feature_decisions": ai_result["feature_decisions"],
                    "missing_outputs": ai_result["missing_outputs"],
                }
            )


def render_evidence_tab(frames: list[MediaFrame], media_summary: list[dict]) -> None:
    st.subheader("3. Expert grading")
    st.write(
        "The Canadian consensus algorithm classifies verified imaging features. AI probabilities, when available, are "
        "kept separate and never become clinical evidence without human confirmation."
    )
    st.info("Complete this form independently of the AI result to preserve an unbiased expert reference grade.")

    all_frames_processed = bool(media_summary) and all(
        bool(item.get("all_frames_processed")) for item in media_summary
    )
    if all_frames_processed:
        st.success(f"Complete sequential decode confirmed for {len(frames)} frames.")
    elif media_summary:
        st.error("At least one source did not decode every reported frame. Final classification is disabled.")

    with st.form("evidence_form"):
        study_code = st.text_input("De-identified study code", placeholder="CUS-0001")
        reader_code = st.text_input(
            "Expert reader code",
            placeholder="R01",
            help="Use a project-issued reader code, not a name or email address.",
        )
        review_round = st.selectbox("Review round", ["Independent read", "Adjudication"])
        a, b = st.columns(2)
        with a:
            has_ga = st.checkbox("Gestational age available")
            ga = st.number_input("Gestational age, weeks", 22.0, 44.0, 28.0, 0.1, disabled=not has_ga)
        with b:
            has_age = st.checkbox("Postnatal age at scan available")
            age = st.number_input("Postnatal age, days", 0.0, 180.0, 3.0, 0.5, disabled=not has_age)

        st.markdown("### Study completeness")
        model_prediction = st.session_state.get("model_prediction") or {}
        plane_counts = model_prediction.get("plane_counts") or {}
        if plane_counts:
            st.caption(
                "Installed-model frame counts: "
                f"coronal {plane_counts.get('coronal', 0)}, sagittal {plane_counts.get('sagittal', 0)}, "
                f"posterior fossa {plane_counts.get('posterior_fossa', 0)}. Confirm coverage from the complete examination."
            )
        coronal_complete = st.checkbox("Complete coronal sweep reviewed and adequate")
        sagittal_complete = st.checkbox("Complete sagittal or parasagittal sweep reviewed and adequate")
        posterior_fossa_complete = st.checkbox("Posterior fossa assessment reviewed and adequate")
        complete_views = coronal_complete and sagittal_complete and posterior_fossa_complete
        serial = st.checkbox("A serial study is available for temporal evolution")

        left_column, right_column = st.columns(2)
        with left_column:
            left = side_form("left")
        with right_column:
            right = side_form("right")

        st.markdown("### White matter injury and posterior fossa")
        wmi_options = {
            "Not assessed": "not_assessed",
            "No ischemic WMI pattern recorded": "none",
            "PVE present for less than 7 days or duration unknown": "pve_under_7_days",
            "Grade 1: PVE persists at least 7 days without cystic evolution": "grade_1",
            "Grade 2: localized frontoparietal cystic evolution": "grade_2",
            "Grade 3: extensive fronto-parieto-occipital periventricular cysts": "grade_3",
            "Grade 4: extensive deep or subcortical white matter cysts": "grade_4",
        }
        cbh_options = {
            "Not assessed": "not_assessed",
            "None recorded": "none",
            "Punctate, 4 mm or smaller": "punctate",
            "Limited, above 4 mm and below one third of hemisphere": "limited",
            "Large, one third or more of hemisphere": "large",
        }
        wmi_label = st.selectbox("Serial ischemic WMI pattern", list(wmi_options))
        cbh_label = st.selectbox("Cerebellar hemorrhage", list(cbh_options))

        st.markdown("### Post-hemorrhagic ventricular dilatation")
        prior = answer_select("Preceding GMH-IVH", "prior_gmh")
        vi97 = answer_select("VI is above the 97th centile", "vi97")
        vi97p4 = answer_select("VI is above the 97th centile plus 4 mm", "vi97p4")

        submitted = st.form_submit_button("Classify with consensus rules", type="primary")

    if submitted:
        model_prediction = st.session_state.get("model_prediction") or {}
        evidence = StudyEvidence(
            study_code=study_code.strip(),
            expert_reader_code=reader_code.strip() or None,
            expert_review_round="adjudication" if review_round == "Adjudication" else "independent",
            postnatal_age_days=float(age) if has_age else None,
            gestational_age_weeks=float(ga) if has_ga else None,
            left=left,
            right=right,
            wmi_pattern=wmi_options[wmi_label],
            cerebellar_hemorrhage=cbh_options[cbh_label],
            prior_gmh_ivh=prior,  # type: ignore[arg-type]
            vi_above_97th=vi97,  # type: ignore[arg-type]
            vi_above_97th_plus_4mm=vi97p4,  # type: ignore[arg-type]
            coronal_views_complete=coronal_complete,
            sagittal_views_complete=sagittal_complete,
            posterior_fossa_views_complete=posterior_fossa_complete,
            complete_required_views=complete_views,
            all_frames_processed=all_frames_processed,
            decoded_frame_count=len(frames),
            serial_study_available=serial,
            model_id=model_prediction.get("model_id"),
            model_version=model_prediction.get("model_version"),
            model_processed_frame_count=model_prediction.get("processed_frame_count"),
            model_plane_counts=model_prediction.get("plane_counts") or {},
            evidence_source="expert",
        )
        st.session_state.study_evidence = evidence
        st.session_state.study_classification = classify_study(evidence)
        manifest, _ = discover_model(MODEL_DIR)
        if model_prediction and manifest is not None:
            ai_result = grade_prediction(
                model_prediction,
                thresholds=manifest.thresholds,
                decision_margin=manifest.decision_margin,
                study_code=evidence.study_code,
                gestational_age_weeks=evidence.gestational_age_weeks,
                postnatal_age_days=evidence.postnatal_age_days,
                all_frames_processed=evidence.all_frames_processed,
                serial_study_available=evidence.serial_study_available,
            )
            st.session_state.ai_consensus_result = ai_result.to_dict()
        st.success("Independent expert consensus classification created.")

    current_classification = st.session_state.get("study_classification")
    if current_classification is not None:
        result = current_classification.to_dict()
        st.divider()
        st.markdown("### Expert Canadian consensus classification")
        if result["classification_status"].startswith("Final"):
            st.success(result["classification_status"])
        else:
            st.warning(result["classification_status"])
        left_result, right_result = st.columns(2)
        with left_result:
            render_side_result("Left hemisphere", result["left"])
        with right_result:
            render_side_result("Right hemisphere", result["right"])
        st.write(f"**White matter injury:** {result['wmi']}")
        st.write(f"**Cerebellar hemorrhage:** {result['cerebellar_hemorrhage']}")
        st.write(f"**PHVD:** {result['phvd']}")
        st.caption("Open Agreement and export to review limitations and download the audit record.")


def render_side_result(title: str, result: dict) -> None:
    st.markdown(
        f"""
        <div class="result-card">
          <div class="tiny">{title.upper()}</div>
          <h3>{result['gmh_ivh']}</h3>
          <p><strong>PVHI:</strong> {result['pvhi']}</p>
          <p><strong>Cystic sequela:</strong> {result['cystic_sequela']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for item in result["reasoning"]:
        st.write(f"- {item}")
    for item in result["warnings"]:
        st.warning(item)


def render_report_tab(media_summary: list[dict]) -> None:
    st.subheader("4. Agreement and export")
    evidence = st.session_state.get("study_evidence")
    classification = st.session_state.get("study_classification")
    if evidence is None or classification is None:
        st.info("Complete the structured evidence form to create a report.")
        return
    ai_result = st.session_state.get("ai_consensus_result")
    agreement_rows = (
        compare_classifications(classification.to_dict(), ai_result["classification"])
        if ai_result
        else []
    )
    reference_rows = cached_reference_labels()
    matched_reference = find_reference_label(reference_rows, evidence.study_code)
    reveal_reference = False
    expert_reference_rows: list[dict] = []
    ai_reference_rows: list[dict] = []
    if matched_reference:
        st.markdown("### Provisional pilot reference")
        st.caption(
            "This single-reader label was supplied by the project lead. Reveal it only after the independent expert form is submitted."
        )
        reveal_reference = st.checkbox("Reveal the provided pilot label and calculate reference agreement")
        if reveal_reference:
            st.dataframe([reference_display_row(matched_reference)], width="stretch", hide_index=True)
            expert_reference_rows = compare_to_reference(
                matched_reference, classification.to_dict(), "expert"
            )
            expert_reference_summary = agreement_summary(expert_reference_rows)
            a_ref, b_ref, c_ref = st.columns(3)
            a_ref.metric("Reference domains compared", expert_reference_summary["domains_compared"])
            b_ref.metric("Expert agreements", expert_reference_summary["domains_agreeing"])
            c_ref.metric(
                "Expert agreement",
                f"{expert_reference_summary['percent_agreement']:.1f}%"
                if expert_reference_summary["percent_agreement"] is not None
                else "Not available",
            )
            st.dataframe(expert_reference_rows, width="stretch", hide_index=True)
            if ai_result:
                ai_reference_rows = compare_to_reference(
                    matched_reference, ai_result["classification"], "ai"
                )
                ai_reference_summary = agreement_summary(ai_reference_rows)
                st.caption(
                    "AI versus reference: "
                    f"{ai_reference_summary['domains_agreeing']}/{ai_reference_summary['domains_compared']} exact agreements."
                )
    elif evidence.study_code:
        st.info("No pilot reference label matched this study code. Use labels such as `Case 6 day 14`.")

    report_reference = matched_reference if reveal_reference else None
    report = build_report(
        evidence,
        classification,
        media_summary,
        model_prediction=st.session_state.get("model_prediction"),
        ai_consensus=ai_result,
        agreement=agreement_summary(agreement_rows) if agreement_rows else None,
        pilot_reference=report_reference,
        expert_reference_agreement=(agreement_summary(expert_reference_rows) if expert_reference_rows else None),
        ai_reference_agreement=(agreement_summary(ai_reference_rows) if ai_reference_rows else None),
    )
    c = report["classification"]
    if c["classification_status"].startswith("Final"):
        st.success(c["classification_status"])
    else:
        st.warning(c["classification_status"])
    st.caption(
        "View coverage: "
        f"coronal {c['view_coverage']['coronal']}, "
        f"sagittal or parasagittal {c['view_coverage']['sagittal_or_parasagittal']}, "
        f"posterior fossa {c['view_coverage']['posterior_fossa']}"
    )
    left_col, right_col = st.columns(2)
    with left_col:
        render_side_result("Left hemisphere", c["left"])
    with right_col:
        render_side_result("Right hemisphere", c["right"])

    st.markdown("### Other consensus domains")
    st.write(f"**White matter injury:** {c['wmi']}")
    st.write(f"**Cerebellar hemorrhage:** {c['cerebellar_hemorrhage']}")
    st.write(f"**PHVD:** {c['phvd']}")
    st.write(f"**Severe preterm brain injury flag:** {c['severe_preterm_brain_injury_flag']}")
    if c["limitations"]:
        st.markdown("### Limitations")
        for item in c["limitations"]:
            st.warning(item)

    if ai_result:
        st.markdown("### AI grading")
        if ai_result["abstained"]:
            st.warning("AI abstained: " + "; ".join(ai_result["abstention_reasons"]))
        ai_classification = ai_result["classification"]
        ai_left, ai_right = st.columns(2)
        with ai_left:
            render_side_result("AI left hemisphere", ai_classification["left"])
        with ai_right:
            render_side_result("AI right hemisphere", ai_classification["right"])
        st.write(f"**AI white matter injury:** {ai_classification['wmi']}")
        st.write(f"**AI cerebellar hemorrhage:** {ai_classification['cerebellar_hemorrhage']}")
        st.write(f"**AI PHVD:** {ai_classification['phvd']}")

        st.markdown("### Expert versus AI agreement")
        summary = agreement_summary(agreement_rows)
        a, b, c_metric = st.columns(3)
        a.metric("Domains compared", summary["domains_compared"])
        b.metric("Exact agreements", summary["domains_agreeing"])
        c_metric.metric("Percent agreement", f"{summary['percent_agreement']:.1f}%")
        st.dataframe(agreement_rows, width="stretch", hide_index=True)
        st.caption(
            "This is within-study exact agreement. Domain-specific Cohen kappa requires multiple independently graded studies."
        )
    else:
        st.info("Run the installed AI model to calculate expert-versus-AI agreement.")

    json_text = json.dumps(report, indent=2)
    markdown_text = report_to_markdown(report)
    model_prediction = st.session_state.get("model_prediction")
    study_csv_text = study_csv(
        evidence.to_dict(),
        classification.to_dict(),
        ai_result,
        model_prediction,
        agreement_rows,
        reference_label=report_reference,
        expert_reference_rows=expert_reference_rows if reveal_reference else None,
        ai_reference_rows=ai_reference_rows if reveal_reference and ai_result else None,
    )
    frame_csv_text = frame_csv(model_prediction)
    a, b, c_download, d_download = st.columns(4)
    with a:
        st.download_button(
            "Download JSON audit record",
            data=json_text,
            file_name=f"{evidence.study_code or 'cus-study'}-report.json",
            mime="application/json",
            width="stretch",
        )
    with b:
        st.download_button(
            "Download Markdown review report",
            data=markdown_text,
            file_name=f"{evidence.study_code or 'cus-study'}-report.md",
            mime="text/markdown",
            width="stretch",
        )
    with c_download:
        st.download_button(
            "Download study raw CSV",
            data=study_csv_text,
            file_name=f"{evidence.study_code or 'cus-study'}-raw-study.csv",
            mime="text/csv",
            width="stretch",
        )
    with d_download:
        st.download_button(
            "Download every-frame AI CSV",
            data=frame_csv_text,
            file_name=f"{evidence.study_code or 'cus-study'}-raw-frames.csv",
            mime="text/csv",
            width="stretch",
            disabled=not bool((model_prediction or {}).get("frame_predictions")),
        )
    with st.expander("Full audit record"):
        st.json(report)
    st.download_button(
        "Download provisional pilot label set CSV",
        data=reference_csv(reference_rows),
        file_name="pilot-reference-labels-v1.csv",
        mime="text/csv",
    )


def sidebar() -> None:
    st.sidebar.title("CUS AI Reader")
    st.sidebar.caption(f"Research prototype v{APP_VERSION}")
    if OFFLINE_MODE:
        st.sidebar.success("Local offline mode")
    manifest, warnings = discover_model(MODEL_DIR)
    if manifest is None:
        st.sidebar.warning("Automatic image model: not installed")
    elif warnings:
        st.sidebar.warning(f"Model: {manifest.model_id} v{manifest.version}, guarded")
    else:
        st.sidebar.success(f"Model: {manifest.model_id} v{manifest.version}")
    st.sidebar.markdown(
        """
        **Consensus source**

        Mohammad K, Scott JN, Leijser LM, et al. Front Pediatr. 2021;9:618236.

        [Open paper](https://doi.org/10.3389/fped.2021.618236)
        """
    )
    st.sidebar.divider()
    if st.sidebar.button("Reset session"):
        for key in list(st.session_state):
            del st.session_state[key]
        st.rerun()


disclaimer_gate()
sidebar()

st.title("CUS AI Reader")
st.caption("Neonatal cranial ultrasound AI, expert grading, agreement, and Canadian consensus classification")
if OFFLINE_MODE:
    st.success(
        "Local offline mode is active. The app is bound to this computer only, Streamlit usage telemetry is disabled, "
        "and uploaded media remains in the active local session."
    )
st.markdown(
    """
    <div class="research-banner">
    Research use only. AI grades are shown separately from independent expert grades. Agreement and raw data exports
    are available for model development, and every output requires review by a qualified clinician.
    </div>
    """,
    unsafe_allow_html=True,
)

media_tab, ai_tab, evidence_tab, report_tab = st.tabs(
    ["Media", "AI grading", "Expert grading", "Agreement and export"]
)
with media_tab:
    frames, media_summary, _ = render_media_tab()
with ai_tab:
    render_model_panel(frames, media_summary)
with evidence_tab:
    render_evidence_tab(frames, media_summary)
with report_tab:
    render_report_tab(media_summary)

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import streamlit as st

from cus_ai.clinical import classify_study
from cus_ai.media import MediaFrame, decode_media, quality_metrics
from cus_ai.model import OnnxFeatureModel, discover_model, prediction_to_json
from cus_ai.reporting import build_report, report_to_markdown
from cus_ai.schemas import SideEvidence, StudyEvidence


APP_VERSION = "0.3.0"
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
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
    verified = st.checkbox("Clinician verified the recorded features", key=f"{side}_verified")
    return SideEvidence(
        side=side,  # type: ignore[arg-type]
        hemorrhage_present=hemorrhage,  # type: ignore[arg-type]
        confined_to_germinal_matrix=confined,  # type: ignore[arg-type]
        intraventricular_blood=intraventricular,  # type: ignore[arg-type]
        ventricular_distension=distension,  # type: ignore[arg-type]
        ahw_mm=float(ahw) if ahw is not None else None,
        adjacent_periventricular_echogenicity=pve,  # type: ignore[arg-type]
        echogenicity_brighter_than_choroid=brighter,  # type: ignore[arg-type]
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
            st.image(frame.image, use_container_width=True)
            st.caption(f"{frame.source_name} | frame {frame.frame_index} | {row['quality_flag']}")
        display_index += 1
    with st.expander("Technical quality table"):
        st.dataframe(metrics_rows, use_container_width=True, hide_index=True)
    return all_frames, media_summary, warnings


def render_model_panel(frames: list[MediaFrame]) -> None:
    st.subheader("Every-frame plane and feature inference")
    manifest, model_warnings = discover_model(MODEL_DIR)
    for warning in model_warnings:
        st.info(warning)
    if manifest is None:
        st.write(
            "The media reader and consensus classifier are active. Image-derived clinical suggestions stay disabled until "
            "a versioned, validated ONNX feature model with coronal and sagittal plane outputs is installed."
        )
        return
    st.write(f"Model: `{manifest.model_id}` version `{manifest.version}`")
    st.write(f"Intended use: {manifest.intended_use}")
    if not frames:
        st.info("Load media before running the installed model.")
        return
    can_run = not model_warnings and (MODEL_DIR / manifest.onnx_file).exists()
    if st.button("Analyze every frame with installed model", disabled=not can_run):
        with st.spinner(f"Analyzing all {len(frames)} frames in order..."):
            try:
                model = OnnxFeatureModel(MODEL_DIR, manifest)
                prediction = model.predict(frames)
                st.session_state.model_prediction = prediction_to_json(prediction)
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
            st.dataframe(frame_rows, use_container_width=True, hide_index=True)
        with st.expander("Study-level model probabilities"):
            st.json(
                {
                    "probabilities": prediction["probabilities"],
                    "probabilities_by_plane": prediction["probabilities_by_plane"],
                }
            )


def render_evidence_tab(frames: list[MediaFrame], media_summary: list[dict]) -> None:
    st.subheader("2. Verify features and classify")
    st.write(
        "The Canadian consensus algorithm classifies verified imaging features. AI probabilities, when available, are "
        "kept separate and never become clinical evidence without human confirmation."
    )
    render_model_panel(frames)
    st.divider()

    all_frames_processed = bool(media_summary) and all(
        bool(item.get("all_frames_processed")) for item in media_summary
    )
    if all_frames_processed:
        st.success(f"Complete sequential decode confirmed for {len(frames)} frames.")
    elif media_summary:
        st.error("At least one source did not decode every reported frame. Final classification is disabled.")

    with st.form("evidence_form"):
        study_code = st.text_input("De-identified study code", placeholder="CUS-0001")
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
            "Not assessed": "not_asseç_v¶‰ËkºwµçG÷—F†öâó2ã"ã÷—F†öâÓ2ã"ãÖVÖ&VBÖÖCcBç¦— ¢G—F†öå6†#SbÒ#D4$TCdDC3sCD#3sdS4#4cSt4S“dc”D3”S“TScƒƒ#CSƒD3ƒ“”c3#T432  ¦gVæ7F–öâvWBÕ&VÖ÷FTf–ÆR°¢&Ò…·7G&–æuÒEW&’Â·7G&–æuÒDFW7F–æF–öâ¢FBÕG—RÔ76VÖ&Ç”æÖR7—7FVÒäæWBä‡GG ¢F6Æ–VçBÒµ7—7FVÒäæWBä‡GGä‡GG6Æ–VçEÓ£¦æWr‚¢G'’°¢F'—FW2ÒF6Æ–VçBävWD'—FT'&”7–æ2‚EW&’’ävWDv—FW"‚’ävWE&W7VÇB‚¢µ7—7FVÒä”òäf–ÆUÓ£¥w&—FTÆÄ'—FW2‚DFW7F–æF–öâÂF'—FW2¢Ğ¢f–æÆÇ’°¢F6Æ–VçBäF—7÷6R‚¢Ğ§Ğ ¤æWrÔ—FVÒÔ—FVÕG—RF—&V7F÷'’Ôf÷&6RÕF‚D÷WGWDF—&V7F÷'’Â÷WBÔçVÆÀ¤æWrÔ—FVÒÔ—FVÕG—RF—&V7F÷'’Ôf÷&6RÕF‚D66†TF—&V7F÷'’Â÷WBÔçVÆÀ¤æWrÔ—FVÒÔ—FVÕG—RF—&V7F÷'’Ôf÷&6RÕF‚G7FvT6öçF–æW"Â÷WBÔçVÆÀ ¦–b‚Öæ÷B…FW7BÕF‚ÔÆ—FW&ÅF‚G—F†öä&6†—fR’’°¢w&—FRÔ†÷7B$F÷væÆöF–ærF†Röff–6–Â—F†öâ2ã"ãVÖ&VFFVB'VçF–ÖRâââ ¢vWBÕ&VÖ÷FTf–ÆRÕW&’G—F†öåW&’ÔFW7F–æF–öâG—F†öä&6†—fP§Ğ ¢F7GVÅ—F†öä†6‚Ò„vWBÔf–ÆT†6‚ÔÆv÷&—F†Ò4„#SbÔÆ—FW&ÅF‚G—F†öä&6†—fR’ä†6€¦–b‚F7GVÅ—F†öä†6‚ÖæRG—F†öå6†#Sb’°¢F‡&÷r%—F†öâ'VçF–ÖR†6‚fW&–f–6F–öâf–ÆVBâ §Ğ ¢G7FvT6öçF–æW%&W6öÇfVBÒµ7—7FVÒä”òåF…Ó£¤vWDgVÆÅF‚‚G7FvT6öçF–æW"¢G7FvU&ö÷E&W6öÇfVBÒµ7—7FVÒä”òåF…Ó£¤vWDgVÆÅF‚‚G7FvU&ö÷B¦–b‚Öæ÷BG7FvU&ö÷E&W6öÇfVBå7F'G5v—F‚‚G7FvT6öçF–æW%&W6öÇfVBÂµ7—7FVÒå7G&–æt6ö×&—6öåÓ£¤÷&F–æÄ–væ÷&T66R’’°¢F‡&÷r%F†R7Fv–ærföÆFW"×W7B&VÖ–â–ç6–FRF†RFVF–6FVBFV×÷&'’'V–ÆBföÆFW"â §Ğ¦–b…FW7BÕF‚ÔÆ—FW&ÅF‚G7FvU&ö÷B’°¢&VÖ÷fRÔ—FVÒÔÆ—FW&ÅF‚G7FvU&ö÷BÕ&V7W'6RÔf÷&6P§Ğ¦–b…FW7BÕF‚ÔÆ—FW&ÅF‚F&6†—fUF‚’°¢&VÖ÷fRÔ—FVÒÔÆ—FW&ÅF‚F&6†—fUF‚Ôf÷&6P§Ğ ¢G'VçF–ÖUF‚Ò¦ö–âÕF‚G7FvU&ö÷B''VçF–ÖR ¢G6—FU6¶vW2Ò¦ö–âÕF‚G'VçF–ÖUF‚$Æ–%Ç6—FR×6¶vW2 ¤æWrÔ—FVÒÔ—FVÕG—RF—&V7F÷'’Ôf÷&6RÕF‚G'VçF–ÖUF‚Â÷WBÔçVÆÀ¤W‡æBÔ&6†—fRÔÆ—FW&ÅF‚G—F†öä&6†—fRÔFW7F–æF–öåF‚G'VçF–ÖUF€¤æWrÔ—FVÒÔ—FVÕG—RF—&V7F÷'’Ôf÷&6RÕF‚G6—FU6¶vW2Â÷WBÔçVÆÀ ¢GF„f–ÆRÒ¦ö–âÕF‚G'VçF–ÖUF‚'—F†öã3"å÷F‚ ¤€¢'—F†öã3"ç¦— ¢"â ¢"ââ ¢$Æ–%Ç6—FR×6¶vW2 ¢&–×÷'B6—FR ¢’Â6WBÔ6öçFVçBÔÆ—FW&ÅF‚GF„f–ÆRÔVæ6öF–ær44” ¥w&—FRÔ†÷7B$–ç7FÆÆ–ærF†RÆö6¶VBÆ–6F–öâFWVæFVæ6–W2–çFòF†R÷'F&ÆR'VçF–ÖRâââ ¢bD'V–ÆE—F†öâÖÒ—–ç7FÆÂ ¢ÒÖF—6&ÆR×—×fW'6–öâÖ6†V6² ¢ÒÖæòÖ6ö×–ÆR ¢Ò×Ww&FR ¢Ò×F&vWBG6—FU6¶vW2 ¢×"„¦ö–âÕF‚G&Wõ&ö÷B'&WV—&VÖVçG2×÷'F&ÆRçG‡B"¦–b‚DÄ5DU„•D4ôDRÖæR’°¢F‡&÷r$FWVæFVæ7’–ç7FÆÆF–öâf–ÆVBâ §Ğ ¢Ff–ÆW5Fô6÷’Ò€¢&ç’ ¢'÷'F&ÆUöÆVæ6†W"ç’ ¢%7F'B5U2’&VFW"æ6ÖB ¢%÷'F&ÆR$TDÔRçG‡B ¢%$TDÔRæÖB ¢'—&ö¦V7BçFöÖÂ ¢'&WV—&VÖVçG2çG‡B ¢'&WV—&VÖVçG2×÷'F&ÆRçG‡B ¢¦f÷&V6‚‚G&VÆF—fUF‚–âFf–ÆW5Fô6÷’’°¢6÷’Ô—FVÒÔÆ—FW&ÅF‚„¦ö–âÕF‚G&Wõ&ö÷BG&VÆF—fUF‚’ÔFW7F–æF–öâ„¦ö–âÕF‚G7FvU&ö÷BG&VÆF—fUF‚§Ğ¦f÷&V6‚‚FföÆFW"–â‚&7W5ö’"Â&ÖöFVÇ2"Â"ç7G&VÖÆ—B"Â&Fö72"’’°¢6÷’Ô—FVÒÔÆ—FW&ÅF‚„¦ö–âÕF‚G&Wõ&ö÷BFföÆFW"’ÔFW7F–æF–öâ„¦ö–âÕF‚G7FvU&ö÷BFföÆFW"’Õ&V7W'6P§Ğ ¥w&—FRÔ†÷7B%'Vææ–ærF†R÷'F&ÆRFWVæFVæ7’6†V6²âââ ¢b„¦ö–âÕF‚G'VçF–ÖUF‚'—F†öâæW†R"’„¦ö–âÕF‚G7FvU&ö÷B'÷'F&ÆUöÆVæ6†W"ç’"’ÒÖ6†V6°¦–b‚DÄ5DU„•D4ôDRÖæR’°¢F‡&÷r%÷'F&ÆRFWVæFVæ7’fW&–f–6F–öâf–ÆVBâ §Ğ ¢FÖæ–fW7EF‚Ò¦ö–âÕF‚G7FvU&ö÷B$Ôä”dU5BÕ4„#SbçG‡B ¢FÖæ–fW7DÆ–æW2ÒvWBÔ6†–ÆD—FVÒÔÆ—FW&ÅF‚G7FvU&ö÷BÔf–ÆRÕ&V7W'6RÀ¢v†W&RÔö&¦V7B²EòägVÆÄæÖRÖæRFÖæ–fW7EF‚ÒÀ¢6÷'BÔö&¦V7BgVÆÄæÖRÀ¢f÷$V6‚Ôö&¦V7B°¢G&VÆF—fRÒEòägVÆÄæÖRå7V'7G&–ær‚G7FvU&ö÷BäÆVæwF‚²’å&WÆ6R‚%Â"Â"ò"¢F†6‚Ò„vWBÔf–ÆT†6‚ÔÆv÷&—F†Ò4„#SbÔÆ—FW&ÅF‚EòägVÆÄæÖR’ä†6€¢"F†6‚G&VÆF—fR ¢Ğ¢FÖæ–fW7DÆ–æW2Â6WBÔ6öçFVçBÔÆ—FW&ÅF‚FÖæ–fW7EF‚ÔVæ6öF–ær44” ¥w&—FRÔ†÷7B$7&VF–ærF†RöffÆ–æR¤•âââ ¤6ö×&W72Ô&6†—fRÔÆ—FW&ÅF‚G7FvU&ö÷BÔFW7F–æF–öåF‚F&6†—fUF‚Ô6ö×&W76–öäÆWfVÂ÷F–ÖÀ¢F&6†—fT†6‚Ò„vWBÔf–ÆT†6‚ÔÆv÷&—F†Ò4„#SbÔÆ—FW&ÅF‚F&6†—fUF‚’ä†6€¢F†6„f–ÆRÒ"F&6†—fUF‚ç6†#SbçG‡B ¢"F&6†—fT†6‚B…7Æ—BÕF‚ÔÆVbF&6†—fUF‚’"Â6WBÔ6öçFVçBÔÆ—FW&ÅF‚F†6„f–ÆRÔVæ6öF–ær44” ¥w&—FRÔ†÷7B%÷'F&ÆR'VæFÆR7&VFVC¢ ¥w&—FRÔ†÷7BF&6†—fUF€¥w&—FRÔ†÷7B%4„#Sc¢F&6†—fT†6‚ 
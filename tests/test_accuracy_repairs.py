import io
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from cus_ai.ai_consensus import grade_prediction
from cus_ai.agreement import agreement_summary, compare_classifications
from cus_ai.clinical import classify_study
from cus_ai.evidence_mapping import side_evidence_from_probabilities
from cus_ai.media import DecodeLimits, _pil_from_array, decode_media
from cus_ai.model import ModelManifest, build_study_prediction, load_model
from cus_ai.schemas import SideEvidence, StudyEvidence
from test_ai_consensus import complete_prediction


def test_upstream_abstention_survives_confident_feature_probabilities():
    prediction = complete_prediction()
    prediction.update(abstained=True, abstention_reasons=["unvalidated model"])
    result = grade_prediction(prediction, serial_study_available=True)
    assert result.abstained
    assert result.reportable_domains == []
    assert result.classification.left.gmh_ivh.startswith("Not assessed")


def test_similarity_cannot_release_a_grade_even_with_high_votes():
    prediction = complete_prediction()
    prediction["aggregation_mode"] = "pilot_similarity_vote"
    result = grade_prediction(prediction, serial_study_available=True)
    assert result.abstained
    assert all(not status.reportable for status in result.domain_status.values())


def test_incomplete_frame_processing_blocks_all_domains():
    result = grade_prediction(complete_prediction(), all_frames_processed=False, serial_study_available=True)
    assert result.abstained


def test_current_hemorrhage_does_not_establish_prior_hemorrhage():
    result = grade_prediction(complete_prediction(), serial_study_available=True, postnatal_age_days=14)
    assert result.evidence.prior_gmh_ivh == "unknown"
    assert not result.domain_status["phvd"].reportable


def test_plane_presence_does_not_certify_a_complete_sweep():
    result = grade_prediction(complete_prediction())
    assert not result.evidence.coronal_views_complete
    assert not result.evidence.complete_required_views


def test_missing_grade_iii_criteria_withhold_only_affected_hemisphere():
    prediction = complete_prediction()
    prediction["probabilities"].pop("left_ahw_above_6_mm")
    result = grade_prediction(prediction)
    assert not result.domain_status["left_gmh_ivh"].reportable
    assert result.domain_status["right_gmh_ivh"].reportable


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_feature_probabilities_are_rejected(value):
    prediction = complete_prediction()
    prediction["probabilities"]["left_intraventricular_blood"] = value
    with pytest.raises(ValueError, match="finite probabilities"):
        grade_prediction(prediction)


def test_missing_compartment_is_not_a_negative_hemorrhage_result():
    side = side_evidence_from_probabilities("left", {"left_intraventricular_blood": 0.02}, {}, 0.05, {})
    assert side.hemorrhage_present == "unknown"


def test_nonventricular_blood_without_localization_is_not_grade_i():
    side = side_evidence_from_probabilities("left", {"left_hemorrhage_present": 0.99, "left_intraventricular_blood": 0.01}, {}, 0.05, {})
    assert side.confined_to_germinal_matrix == "unknown"


def test_inhomogeneous_white_matter_is_not_dismissed_as_physiologic():
    left = SideEvidence(side="left", hemorrhage_present="no", adjacent_periventricular_echogenicity="yes", echogenicity_brighter_than_choroid="no", echogenicity_inhomogeneous="yes")
    result = classify_study(StudyEvidence(study_code="synthetic", left=left))
    assert "ischemic injury" in result.left.pvhi
    unknown = classify_study(StudyEvidence(study_code="synthetic", left=replace(left, echogenicity_inhomogeneous="unknown")))
    assert unknown.left.pvhi == "Indeterminate"


def test_grade_ii_when_distension_is_absent_even_if_ahw_unknown():
    left = SideEvidence(side="left", hemorrhage_present="yes", confined_to_germinal_matrix="no", intraventricular_blood="yes", ventricular_distension="no")
    assert classify_study(StudyEvidence(study_code="synthetic", left=left)).left.gmh_ivh == "Grade II GMH-IVH"


def test_withheld_outputs_do_not_count_as_agreements():
    result = grade_prediction(complete_prediction(), all_frames_processed=False).classification.to_dict()
    summary = agreement_summary(compare_classifications(result, result))
    assert summary["domains_compared"] == 0
    assert summary["percent_agreement"] is None


def test_frame_output_count_mismatch_is_rejected():
    from cus_ai.media import MediaFrame
    manifest = ModelManifest("test", "1", "test.onnx", (32, 32), ["plane_coronal"])
    frames = [MediaFrame("synthetic", 0, Image.new("L", (32, 32)), "image")]
    with pytest.raises(ValueError, match="every supplied frame"):
        build_study_prediction(manifest, frames, np.zeros((0, 1)))


def test_bad_model_checksum_blocks_loading(tmp_path):
    (tmp_path / "test.onnx").write_bytes(b"not verified weights")
    manifest = ModelManifest("test", "1", "test.onnx", (32, 32), [], onnx_sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        load_model(tmp_path, manifest)


def test_dicom_uint8_intensities_are_preserved():
    pixels = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    assert np.array_equal(np.asarray(_pil_from_array(pixels).convert("L")), pixels)


def test_dicom_resize_updates_mm_per_pixel_and_narrow_multiframe_is_not_rgb():
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, UltrasoundMultiFrameImageStorage, generate_uid
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = UltrasoundMultiFrameImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns, ds.NumberOfFrames = 1024, 4, 2
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit, ds.PixelRepresentation = 7, 0
    ds.PixelSpacing = [0.1, 0.2]
    ds.PixelData = np.full((2, 1024, 4), 80, dtype=np.uint8).tobytes()
    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    result = decode_media("synthetic.dcm", buffer.getvalue())
    assert len(result.frames) == 2
    assert result.frames[0].image.size == (2, 512)
    assert result.frames[0].pixel_spacing_mm == pytest.approx((0.2, 0.4))


def test_video_budget_rejects_instead_of_sampling(tmp_path):
    import cv2
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (320, 320))
    if not writer.isOpened():
        pytest.skip("MJPG encoder unavailable")
    for _ in range(7):
        writer.write(np.full((320, 320, 3), 80, dtype=np.uint8))
    writer.release()
    with pytest.raises(ValueError, match="every-frame"):
        decode_media(path.name, path.read_bytes(), DecodeLimits(max_frames=3))
